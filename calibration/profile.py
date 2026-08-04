"""Spatial-masked activation calibration.

The per-channel threshold theta_c is the top-p percentile of that channel's
post-activation values, measured over a handful of batches. Two details matter:

Spatial masking. Zero-padded convolutions produce systematically inflated
responses in the outermost pixels. Including that border in the percentile
pushes every threshold up and starves the network of spikes, so `crop_margin`
pixels are dropped from each side before the statistic is taken.

Bounded memory. Storing every activation for every batch is hundreds of
megabytes for a 1280-channel layer. Each layer instead keeps a per-channel
reservoir of at most `max_samples_per_channel` values, large enough for a
stable 99th percentile and small enough to fit comfortably in RAM.
"""

import numpy as np
import torch

from models.snn import ACTIVATION_TYPES


class ActivationProfiler:
    """Collects a bounded per-channel sample of activations via forward hooks."""

    def __init__(self, hook_root, crop_margin=2, max_samples_per_channel=2048,
                 num_batches=20, hook_types=ACTIVATION_TYPES):
        self.hook_root = hook_root
        self.crop_margin = crop_margin
        self.max_samples_per_channel = max_samples_per_channel
        # Split the per-channel budget evenly across the calibration batches.
        self.per_batch_budget = max(
            1, int(np.ceil(max_samples_per_channel / max(num_batches, 1))))
        self.hook_types = hook_types
        self.samples = {}
        self.channels = {}
        self._handles = []

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or output.dim() != 4:
                return
            # clone(): in-place activations alias their input, and .cpu() is a
            # no-op on a CPU tensor, so without this the stored view can be
            # mutated by a later layer before it is read.
            values = output.detach().float().cpu().clone()

            margin = self.crop_margin
            _b, channels, height, width = values.shape
            if margin > 0 and height > 2 * margin and width > 2 * margin:
                values = values[:, :, margin:-margin, margin:-margin]

            flat = values.permute(1, 0, 2, 3).reshape(channels, -1)
            if flat.shape[1] > self.per_batch_budget:
                idx = torch.randperm(flat.shape[1])[:self.per_batch_budget]
                flat = flat[:, idx]

            self.samples.setdefault(name, []).append(flat.numpy())
            self.channels[name] = channels
        return hook

    def __enter__(self):
        for name, module in self.hook_root.named_modules():
            if isinstance(module, self.hook_types):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def thresholds(self, percentile=99.0):
        """Per-channel top-p thresholds: {layer_name: ndarray[C]}."""
        result = {}
        for name, chunks in self.samples.items():
            stacked = np.concatenate(chunks, axis=1)  # [C, N]
            theta = np.percentile(stacked, percentile, axis=1)
            # A channel that never activates yields theta = 0, which would make
            # the step `x >= 0` fire on every (non-negative) input. Substitute
            # the channel maximum, then a tiny floor, so a dead channel stays
            # silent instead of saturating.
            fallback = stacked.max(axis=1)
            theta = np.where(theta > 0, theta, fallback)
            theta = np.where(theta > 0, theta, 1e-6)
            result[name] = theta.astype(np.float32)
        return result


@torch.no_grad()
def collect_profiles(forward_model, loader, device, num_batches=20,
                     hook_root=None, crop_margin=2,
                     max_samples_per_channel=2048, verbose=True):
    """Stream `num_batches` through the ANN and return an ActivationProfiler.

    Args:
        forward_model: the module actually called on each batch.
        hook_root: where activations are hooked and *named*. Defaults to
            `forward_model`. Pass a sub-module (e.g. the SSD spiking trunk) to
            calibrate only part of the network; the names produced here are the
            same names `convert_to_snn` expects.
    """
    hook_root = hook_root if hook_root is not None else forward_model

    was_training = forward_model.training
    forward_model.eval()

    profiler = ActivationProfiler(
        hook_root, crop_margin=crop_margin,
        max_samples_per_channel=max_samples_per_channel,
        num_batches=num_batches)

    with profiler:
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            forward_model(images)
            if verbose and (i + 1) % 5 == 0:
                print(f'  calibration batch {i + 1}/{num_batches}')

    forward_model.train(was_training)

    if not profiler.samples:
        raise RuntimeError(
            'Calibration captured no activations. Either the loader is empty '
            'or the model has no nn.ReLU/nn.ReLU6 layers under the hook root '
            '(torchvision MobileNetV2 uses ReLU6 -- check ACTIVATION_TYPES).')
    return profiler


def compute_channel_thresholds(profiler_or_profiles, percentile=99.0,
                               crop_margin=2):
    """Per-channel thresholds from a profiler, or from raw captured tensors.

    Accepts an `ActivationProfiler` (the fast path) or a
    `{layer_name: [tensors]}` dict, which is the shape the reference
    implementation produces.
    """
    if isinstance(profiler_or_profiles, ActivationProfiler):
        return profiler_or_profiles.thresholds(percentile)

    thresholds = {}
    for name, tensors in profiler_or_profiles.items():
        stacked = torch.cat(tensors, dim=0)          # [B, C, H, W]
        _b, channels, height, width = stacked.shape
        if crop_margin > 0 and height > 2 * crop_margin and width > 2 * crop_margin:
            stacked = stacked[:, :, crop_margin:-crop_margin,
                              crop_margin:-crop_margin]
        flat = stacked.permute(1, 0, 2, 3).reshape(channels, -1).numpy()
        theta = np.percentile(flat, percentile, axis=1)
        theta = np.where(theta > 0, theta, flat.max(axis=1))
        thresholds[name] = np.where(theta > 0, theta, 1e-6).astype(np.float32)
    return thresholds


def summarize_thresholds(thresholds, max_rows=8):
    """Human-readable digest, for logs and the notebook."""
    lines = [f'Calibrated {len(thresholds)} activation layer(s):']
    for i, (name, theta) in enumerate(thresholds.items()):
        if i >= max_rows:
            lines.append(f'  ... and {len(thresholds) - max_rows} more')
            break
        lines.append(
            f'  {name:<34s} C={len(theta):<5d} '
            f'theta min/mean/max = {theta.min():.4f} / {theta.mean():.4f} / '
            f'{theta.max():.4f}')
    return '\n'.join(lines)


def save_thresholds(thresholds, path):
    np.savez_compressed(path, **{k: v for k, v in thresholds.items()})
    return path


def load_thresholds(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
