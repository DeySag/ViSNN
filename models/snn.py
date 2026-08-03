import torch
import torch.nn as nn


class StrictT1SFN(nn.Module):
    """Single-timestep Scale-and-Fire neuron with per-channel thresholds."""

    def __init__(self, thresholds):
        super().__init__()
        self.register_buffer(
            'thresholds',
            torch.tensor(thresholds, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward(self, x):
        spikes = (x >= self.thresholds).float()
        return spikes * self.thresholds


def convert_to_snn(module, channel_thresholds, device, prefix=''):
    """Recursively swap nn.ReLU for StrictT1SFN using calibrated thresholds."""
    for name, child in module.named_children():
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.ReLU):
            if full_name in channel_thresholds:
                setattr(module, name, StrictT1SFN(channel_thresholds[full_name]).to(device))
        else:
            convert_to_snn(child, channel_thresholds, device, full_name)