import torch.nn as nn

import torchvision.models as models


def make_snn_ready(module):
    """Recursively replace ReLU / ReLU6 with nn.ReLU(inplace=False)."""
    for child_name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(module, child_name, nn.ReLU(inplace=False))
        else:
            make_snn_ready(child)


def get_mobilenetv2_backbone(device):
    """Procure a pretrained, SNN-ready, frozen-baseline MobileNetV2."""
    model = models.mobilenet_v2(weights='DEFAULT')
    make_snn_ready(model)
    model.eval()
    return model.to(device)