import torch
import torch.nn as nn


class SFNNeuron(nn.Module):
    """Scale-and-Fire Neuron (SFN) with per-channel thresholds.

    Reference: "One-Timestep is Enough: Achieving High-performance ANN-to-SNN
    Conversion via Scale-and-Fire Neurons" (arXiv:2510.23383).

    Single-timestep output (o = lambda*theta * G_{lambda*theta}(h)):
      - fire_fn='binary': G is a step at the effective threshold (classic S&F).
      - fire_fn='mtn':     G is the multi-threshold clip/floor quantization.

    Multi-timestep (t > 1): membrane accumulation + reset-by-subtraction over
    duplicated inputs; the network-level loop averages per-step outputs (rate).
    """

    def __init__(self, thresholds, lambda_=1.0, fire_fn='binary',
                 timesteps=1, n_levels=8):
        super().__init__()
        self.register_buffer(
            'thresholds',
            torch.tensor(thresholds, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.lambda_ = lambda_
        self.fire_fn = fire_fn
        self.timesteps = timesteps
        self.n_levels = n_levels
        self.reset_membrane()

    def reset_membrane(self):
        self.membrane = None

    def effective_threshold(self):
        return self.thresholds * self.lambda_

    def _single_step(self, x):
        threshold = self.effective_threshold()
        if self.fire_fn == 'binary':
            fired = (x >= threshold).float()
            return fired * threshold
        if self.fire_fn == 'mtn':
            levels = (x / threshold.clamp(min=1e-6)).floor().clamp(0, self.n_levels)
            return levels * threshold
        raise ValueError(f'Unknown fire_fn: {self.fire_fn}')

    def _accumulate_fire(self, x):
        threshold = self.effective_threshold()
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)
        self.membrane = self.membrane + x
        fired = (self.membrane >= threshold).float()
        self.membrane = self.membrane - fired * threshold
        return fired * threshold

    def forward(self, x):
        if self.timesteps == 1:
            return self._single_step(x)
        return self._accumulate_fire(x)


def convert_to_snn(module, channel_thresholds, device, lambda_=1.0,
                   fire_fn='binary', timesteps=1, n_levels=8, prefix=''):
    """Recursively swap nn.ReLU for SFNNeuron using calibrated thresholds."""
    for name, child in module.named_children():
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.ReLU):
            if full_name in channel_thresholds:
                setattr(module, name, SFNNeuron(
                    thresholds=channel_thresholds[full_name],
                    lambda_=lambda_,
                    fire_fn=fire_fn,
                    timesteps=timesteps,
                    n_levels=n_levels,
                ).to(device))
        else:
            convert_to_snn(child, channel_thresholds, device, lambda_,
                           fire_fn, timesteps, n_levels, full_name)


def set_lambda(module, lambda_):
    """Set the scaling factor on every SFNNeuron in the module (recursive)."""
    for child in module.modules():
        if isinstance(child, SFNNeuron):
            child.lambda_ = lambda_


def reset_spiking_state(module):
    """Zero the membrane potential of every SFNNeuron in the module."""
    for child in module.modules():
        if isinstance(child, SFNNeuron):
            child.reset_membrane()