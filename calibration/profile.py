import numpy as np
import torch
import torch.nn as nn


def collect_profiles(model, loader, device, num_batches, hook_target=nn.ReLU):
    """Capture layer activations via forward hooks over a multi-batch pass."""
    profiles = {}

    def make_hook(layer_name):
        def hook(module, inp, out):
            if layer_name not in profiles:
                profiles[layer_name] = []
            profiles[layer_name].append(out.detach().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, hook_target):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images.to(device))

    for handle in handles:
        handle.remove()
    return profiles


def compute_channel_thresholds(profiles, percentile=99.0, crop_margin=2):
    """Per-channel thresholds computed on the center-cropped spatial core."""
    thresholds = {}
    for layer_name, tensors in profiles.items():
        stacked = torch.cat(tensors, dim=0)  # [B_total, C, H, W]
        C, H, W = stacked.shape[1], stacked.shape[2], stacked.shape[3]
        theta = np.zeros(C)
        for c in range(C):
            channel_map = stacked[:, c, :, :]
            if H > crop_margin * 2 and W > crop_margin * 2:
                safe_core = channel_map[:, crop_margin:-crop_margin, crop_margin:-crop_margin]
            else:
                safe_core = channel_map
            theta[c] = np.percentile(safe_core.numpy(), percentile)
        thresholds[layer_name] = theta
    return thresholds