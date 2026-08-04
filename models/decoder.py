"""Continuous depth decoder.

The decoder is the only trainable component of the depth track. It maps the
frozen spiking encoder's [B, C, 7, 7] feature map up 32x to [B, 1, 224, 224]
metric depth.

A bare ReLU on the final convolution can latch off: once a unit's pre-activation
is negative everywhere its gradient is identically zero and that pixel never
recovers. The final bias is therefore initialised positive so training starts
inside the active region, and 'softplus' is available as a drop-in activation
that cannot die.
"""

import torch
import torch.nn as nn

import config


def _output_activation(kind):
    if kind == 'relu':
        return nn.ReLU(inplace=False)
    if kind == 'softplus':
        return nn.Softplus(beta=1.0)
    if kind in ('linear', 'none', None):
        return nn.Identity()
    raise ValueError(f'Unknown output activation: {kind!r}')


class SimpleDepthDecoder(nn.Module):
    """Bilinear upsampling decoder: [B, in_ch, 7, 7] -> [B, 1, 224, 224].

    Args:
        encoder: the (frozen, spiking) feature extractor, held as a submodule
            so the decoder is a single end-to-end callable; only `self.head`
            carries trainable parameters.
        in_channels: encoder output channels (1280 MobileNetV2 / 2048 ResNet50).
        output_activation: 'relu' | 'softplus' | 'linear'.
        init_bias: initial bias of the final conv, in metres.
    """

    def __init__(self, encoder, in_channels=1280, output_activation='relu',
                 init_bias=5.0, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.output_activation_kind = output_activation

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.activation = _output_activation(output_activation)

        self._init_weights(init_bias)

    def _init_weights(self, init_bias):
        for module in self.head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        final_conv = [m for m in self.head.modules()
                      if isinstance(m, nn.Conv2d)][-1]
        # Small weights and a positive bias yield an initial prediction of
        # roughly uniform `init_bias` metres, inside the data distribution and
        # keeping the output ReLU alive.
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-2)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, float(init_bias))

    def train(self, mode=True):
        """Train the head but keep the encoder pinned to eval().

        A plain `model.train()` would flip the frozen encoder's BatchNorm layers
        to batch-statistics mode, so the activation distribution would no longer
        match what the SNN thresholds were calibrated against.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @property
    def trainable_parameters(self):
        """Exactly the tensors the optimizer should see."""
        return self.head.parameters()

    def forward_features(self, x):
        return self.encoder(x)

    def decode(self, features):
        return self.activation(self.head(features))

    def forward(self, x):
        return self.decode(self.encoder(x))


def build_depth_model(encoder, in_channels, output_activation='relu',
                      init_bias=5.0, device=None):
    model = SimpleDepthDecoder(encoder, in_channels=in_channels,
                               output_activation=output_activation,
                               init_bias=init_bias)
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def sanity_check_decoder(model, input_size=config.INPUT_SIZE, device='cpu'):
    """Assert the decoder really restores the input resolution."""
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    out = model(dummy)
    expected = (1, 1, input_size, input_size)
    if tuple(out.shape) != expected:
        raise RuntimeError(
            f'Decoder output {tuple(out.shape)} != expected {expected}. '
            'Check the Upsample scale factors against the encoder stride.')
    model.train(was_training)
    return tuple(out.shape)
