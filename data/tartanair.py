import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class TartanAirDataset(Dataset):
    """TartanAir depth dataset (exact float depth from .npy, PNG fallback)."""

    def __init__(self, root_dir, transform=None, mode='train', val_fraction=0.1):
        self.root_dir = root_dir
        self.transform = transform

        image_dir = os.path.join(root_dir, 'image_left')
        depth_dir = os.path.join(root_dir, 'depth_left')

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, 'image_left', '**', '*.png'), recursive=True))
        self.depth_paths = sorted(glob.glob(os.path.join(depth_dir, 'depth_left', '**', '*.npy'), recursive=True))

        if len(self.depth_paths) == 0:
            self.depth_paths = sorted(glob.glob(os.path.join(depth_dir, 'depth_left', '**', '*.png'), recursive=True))

        assert len(self.image_paths) == len(self.depth_paths), \
            f"Image/depth mismatch: {len(self.image_paths)} vs {len(self.depth_paths)}"

        n_val = int(len(self.image_paths) * val_fraction)
        if mode == 'train':
            self.image_paths = self.image_paths[:-n_val]
            self.depth_paths = self.depth_paths[:-n_val]
        else:
            self.image_paths = self.image_paths[-n_val:]
            self.depth_paths = self.depth_paths[-n_val:]

        print(f'TartanAir {mode} set: {len(self.image_paths)} samples.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        depth_path = self.depth_paths[idx]
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.array(Image.open(depth_path), dtype=np.float32) / 256.0

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
            depth_tensor = align_depth_target(depth_tensor, image.shape[-2:])

        return image, depth_tensor


def align_depth_target(depth_tensor, output_size):
    """Resize the depth tensor so it stays aligned with the image spatial size."""
    return F.interpolate(depth_tensor.unsqueeze(0), size=output_size,
                         mode='bilinear', align_corners=False).squeeze(0)