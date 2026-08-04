"""Scale-and-Fire spiking neuron and ANN-to-SNN conversion.

Replaces every activation in a pretrained CNN backbone with a single-timestep
spiking neuron:

    o = theta_eff * 1[x >= theta_eff],      theta_eff = lambda * theta_c

`theta_c` is the per-channel threshold obtained by spatial-masked calibration
(see calibration.profile). Emitting `theta_eff` rather than a bare 1.0
preserves the activation magnitude the following convolution expects, which is
what makes T=1 conversion viable.

The step function has essentially zero gradient, so the converted backbone is
frozen and pinned to eval(); gradients only flow through the continuous task
heads. `ACTIVATION_TYPES` covers both `nn.ReLU` and `nn.ReLU6`, which torchvision
uses in MobileNetV2 and which subclasses `nn.Hardtanh` rather than `nn.ReLU`.
"""

import torch
import torch.nn as nn

import config

# Activations the converter handles. ReLU6 subclasses Hardtanh, hence the
# explicit pair rather than a single isinstance check.
ACTIVATION_TYPES = (nn.ReLU, nn.ReLU6)


class StrictT1SFN(nn.Module):
    """Single-timestep Scale-and-Fire neuron with per-channel thresholds.

    Args:
        channel_thresholds: 1-D array of length C of per-channel thresholds
            theta_c.
        lambda_: global scaling factor applied to every threshold.
        fire_fn: 'binary' -> strict step (fires one spike or none);
            'mtn' -> multi-threshold, floor-quantized to ``n_levels``.
        timesteps: T. T=1 uses the closed-form step; T>1 accumulates a membrane
            potential that resets by subtraction, and the network loop in
            `spiking_forward` averages the per-step outputs.
        n_levels: quantization levels for fire_fn='mtn'.
    """

    def __init__(self, channel_thresholds, lambda_=config.LAMBDA,
                 fire_fn=config.FIRE_FN, timesteps=config.TIMESTEPS,
                 n_levels=config.N_LEVELS):
        super().__init__()
        thresholds = torch.as_tensor(channel_thresholds, dtype=torch.float32)
        thresholds = thresholds.flatten()
        # A dead channel (theta=0) would make the step fire everywhere, since
        # post-ReLU tensors satisfy x >= 0. Floor it so dead channels stay silent.
        thresholds = thresholds.clamp(min=1e-6)
        # [1, C, 1, 1] so it broadcasts against [B, C, H, W].
        self.register_buffer('thresholds', thresholds.view(1, -1, 1, 1))

        if fire_fn not in ('binary', 'mtn'):
            raise ValueError(f'Unknown fire_fn: {fire_fn!r}')
        self.lambda_ = float(lambda_)
        self.fire_fn = fire_fn
        self.timesteps = int(timesteps)
        self.n_levels = int(n_levels)

        self.membrane = None
        # Spike-rate telemetry, consumed by validation/energy reporting.
        self.track_spikes = False
        self.register_buffer('spike_count', torch.zeros(()), persistent=False)
        self.register_buffer('element_count', torch.zeros(()), persistent=False)

    # ------------------------------------------------------------------
    @property
    def num_channels(self):
        return self.thresholds.numel()

    def effective_threshold(self, ref=None):
        """lambda * theta, broadcast to the shape of ``ref``."""
        theta = self.thresholds * self.lambda_
        if ref is not None and ref.dim() != 4:
            shape = [1] * ref.dim()
            shape[1 if ref.dim() > 1 else 0] = theta.numel()
            theta = theta.reshape(shape)
        return theta

    def reset_membrane(self):
        self.membrane = None

    def reset_stats(self):
        self.spike_count.zero_()
        self.element_count.zero_()

    @property
    def spike_rate(self):
        """Fraction of neuron-steps that fired since the last reset_stats()."""
        if self.element_count.item() == 0:
            return 0.0
        return (self.spike_count / self.element_count).item()

    # ------------------------------------------------------------------
    def _record(self, fired):
        if self.track_spikes:
            with torch.no_grad():
                self.spike_count += (fired > 0).sum().float()
                self.element_count += fired.numel()

    def _single_step(self, x):
        threshold = self.effective_threshold(x)
        if self.fire_fn == 'binary':
            fired = (x >= threshold).to(x.dtype)
            self._record(fired)
            return fired * threshold
        # 'mtn': floor(x / theta) clipped to [0, n_levels].
        levels = torch.floor(x / threshold.clamp(min=1e-6))
        levels = levels.clamp(0, self.n_levels)
        self._record(levels)
        return levels * threshold

    def _accumulate_fire(self, x):
        threshold = self.effective_threshold(x)
        if self.membrane is None or self.membrane.shape != x.shape:
            self.membrane = torch.zeros_like(x)
        self.membrane = self.membrane + x
        fired = (self.membrane >= threshold).to(x.dtype)
        # Reset by subtraction, not to zero: the residual charge carries into
        # the next timestep, which is what makes the rate code converge to the
        # ANN activation as T grows.
        self.membrane = self.membrane - fired * threshold
        self._record(fired)
        return fired * threshold

    def forward(self, x):
        if self.timesteps == 1:
            return self._single_step(x)
        return self._accumulate_fire(x)

    def extra_repr(self):
        return (f'channels={self.num_channels}, lambda={self.lambda_}, '
                f'fire_fn={self.fire_fn}, T={self.timesteps}')


# ---------------------------------------------------------------------------
# Conversion surgery
# ---------------------------------------------------------------------------
def convert_to_snn(module, channel_thresholds, device=None,
                   lambda_=config.LAMBDA, fire_fn=config.FIRE_FN,
                   timesteps=config.TIMESTEPS, n_levels=config.N_LEVELS,
                   prefix='', strict=False):
    """Recursively replace activations with `StrictT1SFN` in place.

    `channel_thresholds` maps each module's dotted name (as reported by
    `named_modules()`) to a per-channel threshold array. Activations without a
    calibrated entry are left untouched unless ``strict`` is set. Returns the
    number of activations replaced.
    """
    replaced = 0
    for name, child in module.named_children():
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(child, ACTIVATION_TYPES):
            if full_name in channel_thresholds:
                neuron = StrictT1SFN(
                    channel_thresholds=channel_thresholds[full_name],
                    lambda_=lambda_,
                    fire_fn=fire_fn,
                    timesteps=timesteps,
                    n_levels=n_levels,
                )
                if device is not None:
                    neuron = neuron.to(device)
                setattr(module, name, neuron)
                replaced += 1
            elif strict:
                raise KeyError(
                    f'No calibrated threshold for activation {full_name!r}. '
                    'Run calibration over a loader that reaches this layer.')
        else:
            replaced += convert_to_snn(
                child, channel_thresholds, device, lambda_, fire_fn,
                timesteps, n_levels, full_name, strict)
    return replaced


def set_lambda(module, lambda_):
    """Set the global scaling factor on every SFN in the tree."""
    for child in module.modules():
        if isinstance(child, StrictT1SFN):
            child.lambda_ = float(lambda_)


def set_timesteps(module, timesteps):
    for child in module.modules():
        if isinstance(child, StrictT1SFN):
            child.timesteps = int(timesteps)
            child.reset_membrane()


def set_fire_fn(module, fire_fn, n_levels=None):
    for child in module.modules():
        if isinstance(child, StrictT1SFN):
            child.fire_fn = fire_fn
            if n_levels is not None:
                child.n_levels = int(n_levels)


def reset_spiking_state(module):
    """Zero every membrane potential. Call before each forward when T > 1."""
    for child in module.modules():
        if isinstance(child, StrictT1SFN):
            child.reset_membrane()


def set_spike_tracking(module, enabled=True, reset=True):
    for child in module.modules():
        if isinstance(child, StrictT1SFN):
            child.track_spikes = enabled
            if reset:
                child.reset_stats()


def spike_report(module):
    """Per-layer and overall spike rates since the last reset."""
    layers, fired, total = {}, 0.0, 0.0
    for name, child in module.named_modules():
        if isinstance(child, StrictT1SFN):
            layers[name] = child.spike_rate
            fired += child.spike_count.item()
            total += child.element_count.item()
    return {'layers': layers, 'overall': (fired / total) if total else 0.0}


def count_spiking_layers(module):
    return sum(1 for m in module.modules() if isinstance(m, StrictT1SFN))


def spiking_forward(model, x, timesteps=1):
    """Run ``model`` over a static input for T timesteps and average outputs.

    For T=1 this is a plain forward. For T>1 the same frame is presented
    repeatedly and the rate-coded outputs are averaged, the standard ANN-to-SNN
    evaluation protocol.
    """
    if timesteps <= 1:
        reset_spiking_state(model)
        return model(x)

    reset_spiking_state(model)
    accumulated = None
    for _ in range(timesteps):
        out = model(x)
        accumulated = out if accumulated is None else accumulated + out
    return accumulated / timesteps


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------
def freeze_module(module, eval_mode=True):
    """Detach a module from training: no gradients, no BatchNorm drift.

    Setting `requires_grad = False` alone is insufficient: a module left in
    train() mode continues updating its BatchNorm running statistics on every
    forward, shifting the distribution the thresholds were calibrated against.
    """
    for param in module.parameters():
        param.requires_grad = False
    for buf_module in module.modules():
        if isinstance(buf_module, nn.modules.batchnorm._BatchNorm):
            buf_module.track_running_stats = False
    if eval_mode:
        module.eval()
    return module


def assert_frozen(module, name='backbone'):
    live = [n for n, p in module.named_parameters() if p.requires_grad]
    if live:
        raise AssertionError(
            f'{name} is not fully frozen; {len(live)} tensor(s) still require '
            f'grad, first: {live[0]}')


def assert_trainable(module, name='head'):
    dead = [n for n, p in module.named_parameters() if not p.requires_grad]
    if dead:
        raise AssertionError(
            f'{name} has frozen gradients; {len(dead)} tensor(s) do not '
            f'require grad, first: {dead[0]}')
