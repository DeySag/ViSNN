"""KITTI depth-prediction dataset.

Indexing uses `os.walk` rather than recursive globbing: on network-mounted
storage (Kaggle, Colab Drive) a `**` glob materialises the whole tree in memory
before returning, whereas `os.walk` yields directory-by-directory. The resulting
(rgb, depth) pair list is cached to JSON so the walk runs once.

Expected layout -- the walk does not depend on the exact nesting, only that
these two patterns appear somewhere under `root_dir`:

    .../<drive>_sync/image_02/data/0000000005.png                     (RGB)
    .../<drive>_sync/proj_depth/groundtruth/image_02/0000000005.png   (depth)

Depth PNGs are 16-bit; metres = pixel / 256.0, and pixel == 0 means "no LiDAR
return here", which every consumer must mask out.
"""

import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
from data.transforms import DepthJointTransform, load_rgb

DRIVE_RE = re.compile(r'\d{4}_\d{2}_\d{2}_drive_\d{4}_sync')
CAMERA_RE = re.compile(r'^image_0\d$')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def _parts(path):
    return os.path.normpath(path).replace('\\', '/').split('/')


def _pair_key(path):
    """Build a (drive, camera, frame) key that a RGB and a depth file share."""
    parts = _parts(path)
    frame = os.path.splitext(parts[-1])[0]

    drive = next((p for p in parts if DRIVE_RE.fullmatch(p)), None)
    camera = next((p for p in reversed(parts[:-1]) if CAMERA_RE.fullmatch(p)),
                  None)

    if drive is None:
        # Fall back to the nearest ancestor that is not a structural folder.
        skip = {'data', 'groundtruth', 'velodyne_raw', 'proj_depth'}
        drive = next(
            (p for p in reversed(parts[:-1])
             if not CAMERA_RE.fullmatch(p) and p not in skip),
            'unknown',
        )
    if camera is None:
        camera = 'image_02'
    return drive, camera, frame


def _is_depth(path):
    lowered = os.path.normpath(path).replace('\\', '/').lower()
    return '/groundtruth/' in lowered or lowered.endswith('/groundtruth')


def _is_rgb(path):
    lowered = os.path.normpath(path).replace('\\', '/').lower()
    return '/data/' in lowered and '/proj_depth/' not in lowered


def build_index(root_dir, cameras=('image_02', 'image_03')):
    """Walk `root_dir` once and return the matched (rgb, depth) path pairs."""
    rgb_map, depth_map = {}, {}

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(IMAGE_EXTS):
                continue
            full = os.path.join(dirpath, filename)
            key = _pair_key(full)
            if key[1] not in cameras:
                continue
            if _is_depth(full):
                depth_map[key] = full
            elif _is_rgb(full):
                rgb_map[key] = full

    pairs = [(rgb_map[k], depth_map[k]) for k in sorted(rgb_map)
             if k in depth_map]
    return pairs


class KITTIDepthDataset(Dataset):
    """Streams (image, depth) pairs from the KITTI depth-prediction set.

    Returns:
        image: float32 [3, 224, 224], ImageNet-normalized.
        depth: float32 [1, 224, 224], metres; 0.0 marks invalid pixels.
    """

    def __init__(self, root_dir=config.KITTI_ROOT, transform=None,
                 split='train', val_fraction=0.1, cameras=('image_02',),
                 max_samples=None, cache_index=True):
        self.root_dir = root_dir
        self.transform = transform or DepthJointTransform(
            augment=(split == 'train'))

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f'KITTI root not found: {root_dir}\n'
                'Download the KITTI depth-prediction set (annotated depth maps '
                '+ the matching raw frames) and point config.KITTI_ROOT at it, '
                'or run with --synthetic to exercise the pipeline without data.'
            )

        pairs = self._load_or_build_index(cameras, cache_index)
        if not pairs:
            raise RuntimeError(
                f'No matched RGB/depth pairs under {root_dir}. Confirm both the '
                'raw `image_0X/data/` frames and the `proj_depth/groundtruth/` '
                'annotations are present -- the annotated archive alone has no '
                'RGB images.'
            )

        # Deterministic drive-disjoint split: whole sequences go to one side so
        # near-duplicate consecutive frames cannot leak train->val.
        drives = sorted({_pair_key(rgb)[0] for rgb, _ in pairs})
        n_val = max(1, int(round(len(drives) * val_fraction)))
        val_drives = set(drives[-n_val:]) if len(drives) > 1 else set()

        if split == 'train':
            selected = [p for p in pairs if _pair_key(p[0])[0] not in val_drives]
        elif split == 'val':
            selected = [p for p in pairs if _pair_key(p[0])[0] in val_drives]
            if not selected:  # single-drive dataset: fall back to a tail split
                cut = int(len(pairs) * (1 - val_fraction))
                selected = pairs[cut:] or pairs[-1:]
        elif split == 'all':
            selected = pairs
        else:
            raise ValueError(f"split must be 'train'|'val'|'all', got {split!r}")

        if max_samples is not None:
            selected = selected[:max_samples]

        self.image_paths = [p[0] for p in selected]
        self.depth_paths = [p[1] for p in selected]

    def _load_or_build_index(self, cameras, cache_index):
        cache_path = os.path.join(self.root_dir, '.visnn_index.json')
        if cache_index and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as handle:
                    cached = json.load(handle)
                if cached.get('cameras') == list(cameras):
                    return [tuple(p) for p in cached['pairs']]
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt or stale cache -> rebuild

        pairs = build_index(self.root_dir, cameras=cameras)
        if cache_index and pairs:
            try:
                with open(cache_path, 'w', encoding='utf-8') as handle:
                    json.dump({'cameras': list(cameras), 'pairs': pairs}, handle)
            except OSError:
                pass  # read-only mount (Kaggle input) -- caching is optional
        return pairs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_rgb(self.image_paths[idx])

        # 16-bit PNG -> metres. Read via numpy so the 16-bit range survives;
        # PIL's 'I;16' mode would otherwise be clipped by a naive convert().
        with Image.open(self.depth_paths[idx]) as depth_png:
            depth = np.array(depth_png, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth / config.KITTI_DEPTH_SCALE

        image_t, depth_t = self.transform(image, depth)
        return image_t, depth_t


def depth_collate(batch):
    """Default-style collate kept explicit for symmetry with the SSD collate."""
    images = torch.stack([item[0] for item in batch], dim=0)
    depths = torch.stack([item[1] for item in batch], dim=0)
    return images, depths
