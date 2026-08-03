import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

import numpy as np


class KITTIDepthDataset(Dataset):
    """KITTI dense depth dataset (16-bit PNG LiDAR annotations)."""

    def __init__(self, root_dir, transform=None, mode='train', val_fraction=0.1):
        self.root_dir = root_dir
        self.transform = transform

        image_dir = os.path.join(root_dir, 'image')
        depth_dir = os.path.join(root_dir, 'depth')

        self.image_paths = self._walk_files(image_dir, ('.png', '.jpg', '.jpeg'))
        self.depth_paths = self._walk_files(depth_dir, ('.png',))

        assert len(self.image_paths) == len(self.depth_paths), \
            f"Image/depth mismatch: {len(self.image_paths)} vs {len(self.depth_paths)}"

        n_val = int(len(self.image_paths) * val_fraction)
        if mode == 'train':
            self.image_paths = self.image_paths[:-n_val]
            self.depth_paths = self.depth_paths[:-n_val]
        else:
            self.image_paths = self.image_paths[-n_val:]
            self.depth_paths = self.depth_paths[-n_val:]

        print(f'KITTI {mode} set: {len(self.image_paths)} samples.')

    @staticmethod
    def _walk_files(root, exts):
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                if fname.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, fname))
        return sorted(paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # KITTI 16-bit PNG depth: meters = pixel_value / 256.0
        depth = np.array(Image.open(self.depth_paths[idx]), dtype=np.float32) / 256.0
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
            depth_tensor = align_depth_target(depth_tensor, image.shape[-2:])

        return image, depth_tensor


def align_depth_target(depth_tensor, output_size):
    """Center-crop the depth tensor so it stays aligned with the image crop."""
    return TF.center_crop(depth_tensor, output_size)