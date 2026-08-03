import torch
import torch.nn as nn

from .snn import reset_spiking_state


class SpikingEncoder(nn.Module):
    """Runs a frozen feature extractor across T timesteps and returns the
    rate-averaged features. T=1 degenerates to a single forward pass.

    For T > 1, the input is duplicated each step so the SFN layers accumulate
    membrane potential; the per-step feature maps are averaged to a spike rate.
    """

    def __init__(self, feature_layers, timesteps=1):
        super().__init__()
        self.feature_layers = feature_layers
        self.timesteps = timesteps

    def forward(self, x):
        reset_spiking_state(self.feature_layers)
        if self.timesteps == 1:
            return self.feature_layers(x)

        outputs = []
        for _ in range(self.timesteps):
            outputs.append(self.feature_layers(x))
        return torch.stack(outputs, dim=0).mean(dim=0)