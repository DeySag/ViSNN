import torch.nn as nn


class SimpleDepthDecoder(nn.Module):
    """Continuous upsampling decoder fed by a (frozen) SNN encoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))