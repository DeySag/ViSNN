"""Synthetic stand-ins for KITTI and COCO.

These allow the entire pipeline -- calibration, SNN conversion, training,
validation, mAP, visualisation -- to be exercised before the real (large)
datasets are downloaded. They emit tensors with exactly the shapes, dtypes and
value conventions of the real datasets and pass through the same transform
objects, so swapping in the real loader changes nothing but the Dataset class.

Deterministic: sample `i` always renders identically, so metrics are
reproducible across runs.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

import config
from data.transforms import DepthJointTransform, DetectionTransform

# A handful of named classes so detection visualisations are legible.
SYNTHETIC_CLASSES = ['__background__', 'box', 'disc', 'bar']


def _rng(seed, idx):
    return np.random.default_rng(seed * 100003 + idx)


class SyntheticDepthDataset(Dataset):
    """KITTI-shaped fake data: wide RGB frames plus sparse metric depth.

    Mimics the properties that actually matter for Track A:
      * KITTI-like 1242x375 aspect ratio, so the CenterCrop path is exercised;
      * depth in metres with a ground-plane gradient plus foreground objects;
      * ~85% of pixels zeroed, matching the sparsity of LiDAR ground truth.
    """

    def __init__(self, length=64, transform=None, split='train',
                 seed=config.SEED, frame_size=(1242, 375), valid_ratio=0.15):
        self.length = length
        self.frame_size = frame_size
        self.valid_ratio = valid_ratio
        self.seed = seed + (0 if split == 'train' else 7919)
        self.transform = transform or DepthJointTransform(
            augment=False)  # keep synthetic runs deterministic

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = _rng(self.seed, idx)
        width, height = self.frame_size

        # Depth: a ground plane receding with image row, 5m at the bottom edge
        # up to DEPTH_MAX near the horizon.
        rows = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
        depth = 5.0 + rows * 55.0
        depth = np.repeat(depth, width, axis=1)

        image = Image.new('RGB', (width, height), color=(90, 110, 130))
        draw = ImageDraw.Draw(image)

        # Sky gradient in the top third, so the RGB carries real structure.
        for row in range(height // 3):
            shade = int(180 - 60 * row / max(height // 3, 1))
            draw.line([(0, row), (width, row)], fill=(shade, shade, 210))

        for _ in range(rng.integers(3, 8)):
            obj_w = int(rng.integers(60, 220))
            obj_h = int(rng.integers(50, 180))
            x0 = int(rng.integers(0, max(width - obj_w, 1)))
            y0 = int(rng.integers(height // 3, max(height - obj_h, height // 3 + 1)))
            x1, y1 = x0 + obj_w, y0 + obj_h
            obj_depth = float(rng.uniform(3.0, 40.0))

            # Nearer objects render brighter -- gives the encoder a genuine
            # appearance/depth correlation to latch onto.
            tone = int(np.clip(255 - obj_depth * 5.0, 40, 255))
            draw.rectangle([x0, y0, x1, y1],
                           fill=(tone, int(tone * 0.6), int(tone * 0.4)))
            depth[y0:y1, x0:x1] = obj_depth

        depth = np.clip(depth, 0.0, config.DEPTH_MAX).astype(np.float32)

        # Sparsify to LiDAR-like coverage: 0.0 means "no return".
        mask = rng.random(depth.shape) < self.valid_ratio
        depth = depth * mask.astype(np.float32)

        return self.transform(image, depth)


class SyntheticDetectionDataset(Dataset):
    """COCO-shaped fake data: rendered shapes with exact ground-truth boxes.

    Boxes are known analytically (they are what was drawn), so a correct
    detector can drive mAP genuinely high -- which makes this a real test of
    the mAP implementation, not just a smoke test.
    """

    def __init__(self, length=64, transform=None, split='train',
                 seed=config.SEED, frame_size=(480, 360), max_objects=4):
        self.length = length
        self.frame_size = frame_size
        self.max_objects = max_objects
        self.seed = seed + (0 if split.startswith('train') else 104729)
        self.transform = transform or DetectionTransform(augment=False)
        self.dense_to_name = dict(enumerate(SYNTHETIC_CLASSES))
        self.num_classes = len(SYNTHETIC_CLASSES)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = _rng(self.seed, idx)
        width, height = self.frame_size

        image = Image.new('RGB', (width, height),
                          color=tuple(int(v) for v in rng.integers(30, 90, 3)))
        draw = ImageDraw.Draw(image)

        boxes, labels = [], []
        for _ in range(int(rng.integers(1, self.max_objects + 1))):
            obj_w = int(rng.integers(40, width // 3))
            obj_h = int(rng.integers(40, height // 3))
            x0 = int(rng.integers(0, width - obj_w))
            y0 = int(rng.integers(0, height - obj_h))
            x1, y1 = x0 + obj_w, y0 + obj_h

            label = int(rng.integers(1, len(SYNTHETIC_CLASSES)))
            colour = [(220, 70, 70), (70, 200, 120), (90, 130, 240)][label - 1]

            if label == 1:
                draw.rectangle([x0, y0, x1, y1], fill=colour)
            elif label == 2:
                draw.ellipse([x0, y0, x1, y1], fill=colour)
            else:
                draw.rectangle([x0, y0, x1, y1], outline=colour, width=6)

            boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])
            labels.append(label)

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        return self.transform(image, boxes_t, labels_t)
