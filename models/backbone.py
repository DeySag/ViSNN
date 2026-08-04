"""Pretrained CNN backbones.

Both tracks share a MobileNetV2 family; the depth track can optionally use
ResNet-50 instead. Weights are downloaded from torchvision on first use, and the
loader falls back to random initialisation when there is no network.

ResNet needs one structural fix before conversion. torchvision's
`Bottleneck.forward` invokes a single `self.relu` module three times, on tensors
with two different channel counts. Per-channel thresholds and forward hooks both
assume one activation module per activation site, so `split_resnet_relus`
rebinds `forward` to use three distinct ReLU modules before calibration runs.
"""

import types
import warnings

import torch
import torch.nn as nn
import torchvision.models as tvm

# Feature-map strides for MobileNetV2 at 224x224 input:
#   features[0:2]  -> 112   features[2:4] -> 56    features[4:7] -> 28
#   features[7:14] -> 14    features[14:] -> 7
MOBILENET_TAP_INDICES = (13, 18)     # 96ch @ 14x14, 1280ch @ 7x7
MOBILENET_TAP_CHANNELS = (96, 1280)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def _load_torchvision(builder, weights_enum, pretrained):
    """Instantiate a torchvision model, tolerating an offline weight cache."""
    if not pretrained:
        return builder(weights=None)
    try:
        return builder(weights=weights_enum.IMAGENET1K_V1)
    except Exception as exc:  # no network / no cached checkpoint
        warnings.warn(
            f'Could not fetch pretrained weights ({type(exc).__name__}: {exc}). '
            'Falling back to random initialisation -- calibration and training '
            'still run, but accuracy will be far from the reported numbers.',
            RuntimeWarning,
        )
        return builder(weights=None)


# ---------------------------------------------------------------------------
# ResNet activation de-sharing
# ---------------------------------------------------------------------------
def _basic_block_forward(self, x):
    identity = x
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    return self.relu2(out)


def _bottleneck_forward(self, x):
    identity = x
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    return self.relu3(out)


def split_resnet_relus(model):
    """Give every ResNet activation site its own module. Returns the model."""
    for module in model.modules():
        if isinstance(module, tvm.resnet.Bottleneck):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.relu3 = nn.ReLU(inplace=False)
            module.forward = types.MethodType(_bottleneck_forward, module)
        elif isinstance(module, tvm.resnet.BasicBlock):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.forward = types.MethodType(_basic_block_forward, module)
    return model


def disable_inplace(model):
    """Turn off in-place activations.

    In-place ReLU aliases its input, so forward hooks during calibration would
    observe post-mutation values. The memory saving is irrelevant at these
    sizes; correctness is not.
    """
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    return model


# ---------------------------------------------------------------------------
# Single-scale backbones (Track A -- depth)
# ---------------------------------------------------------------------------
class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 feature extractor: [B, 3, 224, 224] -> [B, 1280, 7, 7]."""

    out_channels = 1280
    out_stride = 32

    def __init__(self, pretrained=True):
        super().__init__()
        net = _load_torchvision(tvm.mobilenet_v2,
                                tvm.MobileNet_V2_Weights, pretrained)
        self.features = disable_inplace(net.features)

    def forward(self, x):
        return self.features(x)


class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor: [B, 3, 224, 224] -> [B, 2048, 7, 7]."""

    out_channels = 2048
    out_stride = 32

    def __init__(self, pretrained=True):
        super().__init__()
        net = _load_torchvision(tvm.resnet50, tvm.ResNet50_Weights, pretrained)
        split_resnet_relus(net)
        disable_inplace(net)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def build_backbone(name='mobilenet_v2', pretrained=True):
    """Return (module, out_channels) for the requested depth backbone."""
    name = name.lower()
    if name in ('mobilenet_v2', 'mobilenetv2', 'mobilenet'):
        backbone = MobileNetV2Backbone(pretrained=pretrained)
    elif name in ('resnet50', 'resnet_50', 'resnet'):
        backbone = ResNet50Backbone(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown backbone {name!r}; expected 'mobilenet_v2' or 'resnet50'")
    return backbone, backbone.out_channels


# ---------------------------------------------------------------------------
# Multi-scale backbone (Track B -- SSD)
# ---------------------------------------------------------------------------
class MobileNetV2SSDBackbone(nn.Module):
    """MobileNetV2 split at two taps, plus SSD extra layers.

    Feature maps produced at 224x224 input:
        14x14 (96ch)  <- features[:14]      the spiking trunk
         7x7 (1280ch) <- features[14:]      the spiking trunk
         4x4 (512ch)  \\
         2x2 (256ch)   >- continuous extra layers, trained with the heads
         1x1 (256ch)  /

    Only the MobileNet part is converted to spikes; the extra layers stay
    continuous because they have no pretrained activation statistics to
    calibrate against.
    """

    feature_channels = (96, 1280, 512, 256, 256)
    feature_sizes = (14, 7, 4, 2, 1)

    def __init__(self, pretrained=True):
        super().__init__()
        net = _load_torchvision(tvm.mobilenet_v2,
                                tvm.MobileNet_V2_Weights, pretrained)
        features = disable_inplace(net.features)

        split = MOBILENET_TAP_INDICES[0] + 1  # 14 -> stride-16 tap boundary
        self.stage1 = nn.Sequential(*list(features.children())[:split])
        self.stage2 = nn.Sequential(*list(features.children())[split:])

        self.extra1 = self._extra_block(1280, 256, 512, stride=2)   # 7 -> 4
        self.extra2 = self._extra_block(512, 128, 256, stride=2)    # 4 -> 2
        self.extra3 = self._extra_block(256, 128, 256, stride=2)    # 2 -> 1

    @staticmethod
    def _extra_block(in_ch, mid_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def spiking_trunk(self):
        """The sub-modules that undergo SNN conversion (excluding the extras).

        A method, not a stored attribute: holding the same stages under a second
        attribute would duplicate them in `named_modules()`/`state_dict()`. The
        returned ModuleList wraps the live stage objects, so converting it
        mutates the real backbone, and its module names ('0.*', '1.*') are
        structural and identical on every call, which is what lets calibration
        and conversion agree.
        """
        return nn.ModuleList([self.stage1, self.stage2])

    def forward(self, x):
        f1 = self.stage1(x)     # [B, 96, 14, 14]
        f2 = self.stage2(f1)    # [B, 1280, 7, 7]
        f3 = self.extra1(f2)    # [B, 512, 4, 4]
        f4 = self.extra2(f3)    # [B, 256, 2, 2]
        f5 = self.extra3(f4)    # [B, 256, 1, 1]
        return [f1, f2, f3, f4, f5]


@torch.no_grad()
def infer_feature_shapes(backbone, input_size=224, device='cpu'):
    """Run one dummy batch to read off real feature-map shapes."""
    was_training = backbone.training
    backbone.eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    out = backbone(dummy)
    if torch.is_tensor(out):
        out = [out]
    shapes = [(t.shape[1], t.shape[2], t.shape[3]) for t in out]
    backbone.train(was_training)
    return shapes
