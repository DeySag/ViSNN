"""Spatial transforms.

The hard requirement is geometric alignment: whatever happens to the RGB frame
must happen identically to its depth target, or the loss compares a pixel to
the wrong pixel. The image/target transforms therefore live in one joint
callable rather than two independent `transforms.Compose` chains.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config


# ---------------------------------------------------------------------------
# Track A -- depth
# ---------------------------------------------------------------------------
class DepthJointTransform:
    """Aligned RGB/depth transform for KITTI.

    Args:
        size: output side length of the square tensor.
        mode: 'crop' -> deterministic CenterCrop (true metric scale preserved);
              'resize' -> squash the full frame into the square (keeps context).
        augment: enable joint random horizontal flip + photometric jitter.
                 Off by default so validation stays deterministic.
    """

    def __init__(self, size=config.INPUT_SIZE, mode=config.KITTI_SPATIAL_MODE,
                 mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD,
                 augment=False):
        if mode not in ('crop', 'resize'):
            raise ValueError(f"mode must be 'crop' or 'resize', got {mode!r}")
        self.size = size
        self.mode = mode
        self.mean = mean
        self.std = std
        self.augment = augment

    def __call__(self, image, depth):
        """image: PIL.Image (RGB). depth: float32 ndarray [H, W] in metres."""
        depth_t = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0)

        # The raw frame and its depth map must start at identical resolution.
        w, h = image.size
        if depth_t.shape[-2:] != (h, w):
            depth_t = TF.resize(
                depth_t, [h, w],
                interpolation=TF.InterpolationMode.NEAREST,
            )

        if self.mode == 'crop':
            # A CenterCrop larger than the source would zero-pad and invent
            # depth, so shrink the frame first when KITTI's 375px height (or a
            # smaller variant) cannot cover the requested square.
            if h < self.size or w < self.size:
                scale = self.size / min(h, w)
                new_h, new_w = int(round(h * scale)), int(round(w * scale))
                image = TF.resize(image, [new_h, new_w],
                                  interpolation=TF.InterpolationMode.BILINEAR)
                depth_t = TF.resize(depth_t, [new_h, new_w],
                                    interpolation=TF.InterpolationMode.NEAREST)
            image = TF.center_crop(image, [self.size, self.size])
            depth_t = TF.center_crop(depth_t, [self.size, self.size])
        else:
            image = TF.resize(image, [self.size, self.size],
                              interpolation=TF.InterpolationMode.BILINEAR)
            # NEAREST only: bilinear would smear the sparse LiDAR returns into
            # the invalid (zero) pixels and fabricate depth.
            depth_t = TF.resize(depth_t, [self.size, self.size],
                                interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                depth_t = TF.hflip(depth_t)
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.mean, self.std)

        # Clamp to the KITTI evaluation range. Zeros stay zero and are treated
        # as "invalid" by the masked loss/metrics downstream.
        depth_t = depth_t.clamp(0.0, config.DEPTH_MAX)
        return image_t, depth_t


def align_depth_target(depth_tensor, size=config.INPUT_SIZE):
    """Standalone CenterCrop for a depth tensor, matching the RGB crop.

    Kept as a module-level helper so the crop can be reapplied outside the
    Dataset (e.g. when re-cropping cached targets).
    """
    return TF.center_crop(depth_tensor, [size, size])


def denormalize(image_tensor, mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD):
    """Undo ImageNet normalization -> [0, 1] tensor ready for matplotlib."""
    mean_t = torch.tensor(mean, device=image_tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=image_tensor.device).view(-1, 1, 1)
    return (image_tensor * std_t + mean_t).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Track B -- detection
# ---------------------------------------------------------------------------
class DetectionTransform:
    """Resize-to-square for SSD, with boxes carried in fractional coordinates.

    Boxes are normalized to [0, 1] *before* the resize, which makes them
    invariant to it -- no box arithmetic is needed for the scale change itself.
    Flips still have to be applied to the boxes explicitly.
    """

    def __init__(self, size=config.INPUT_SIZE, mean=config.IMAGENET_MEAN,
                 std=config.IMAGENET_STD, augment=False):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = augment

    def __call__(self, image, boxes, labels):
        """image: PIL RGB. boxes: [N, 4] normalized xyxy. labels: [N] int64."""
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            if boxes.numel():
                flipped = boxes.clone()
                flipped[:, 0] = 1.0 - boxes[:, 2]
                flipped[:, 2] = 1.0 - boxes[:, 0]
                boxes = flipped

        if self.augment and random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        image = TF.resize(image, [self.size, self.size],
                          interpolation=TF.InterpolationMode.BILINEAR)
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.mean, self.std)
        return image_t, boxes, labels


def load_rgb(path):
    """Open an image as RGB, tolerating greyscale and CMYK sources."""
    with Image.open(path) as img:
        return img.convert('RGB')
