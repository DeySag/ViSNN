"""DataLoader construction.

Central place for batch size, worker count, pinning and the real-vs-synthetic
switch. `num_workers > 0` on Windows re-imports this module in every worker
process, which is why the entrypoints guard themselves with
`if __name__ == '__main__'`.
"""

import torch
from torch.utils.data import DataLoader

import config
from data.coco import COCODetectionDataset, detection_collate
from data.kitti import KITTIDepthDataset, depth_collate
from data.synthetic import SyntheticDepthDataset, SyntheticDetectionDataset
from data.transforms import DepthJointTransform, DetectionTransform


def _loader_kwargs(num_workers, device):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': device is not None and device.type == 'cuda',
        'drop_last': False,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


# ---------------------------------------------------------------------------
# Track A -- depth
# ---------------------------------------------------------------------------
def build_depth_datasets(root=config.KITTI_ROOT, synthetic=False,
                         spatial_mode=config.KITTI_SPATIAL_MODE,
                         augment=True, max_train=None, max_val=None):
    train_tf = DepthJointTransform(mode=spatial_mode, augment=augment)
    val_tf = DepthJointTransform(mode=spatial_mode, augment=False)

    if synthetic:
        train_ds = SyntheticDepthDataset(length=max_train or 64,
                                         transform=train_tf, split='train')
        val_ds = SyntheticDepthDataset(length=max_val or 16,
                                       transform=val_tf, split='val')
        return train_ds, val_ds

    train_ds = KITTIDepthDataset(root_dir=root, transform=train_tf,
                                 split='train', max_samples=max_train)
    val_ds = KITTIDepthDataset(root_dir=root, transform=val_tf,
                               split='val', max_samples=max_val)
    return train_ds, val_ds


def build_depth_loaders(root=config.KITTI_ROOT, synthetic=False,
                        batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS, device=None,
                        spatial_mode=config.KITTI_SPATIAL_MODE, augment=True,
                        max_train=None, max_val=None):
    train_ds, val_ds = build_depth_datasets(
        root=root, synthetic=synthetic, spatial_mode=spatial_mode,
        augment=augment, max_train=max_train, max_val=max_val)

    kwargs = _loader_kwargs(num_workers, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=depth_collate, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=depth_collate, **kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Track B -- detection
# ---------------------------------------------------------------------------
def build_detection_datasets(root=config.COCO_ROOT, synthetic=False,
                             augment=True, max_train=None, max_val=None):
    train_tf = DetectionTransform(augment=augment)
    val_tf = DetectionTransform(augment=False)

    if synthetic:
        train_ds = SyntheticDetectionDataset(length=max_train or 64,
                                             transform=train_tf, split='train')
        val_ds = SyntheticDetectionDataset(length=max_val or 16,
                                           transform=val_tf, split='val')
        return train_ds, val_ds

    train_ds = COCODetectionDataset(root_dir=root,
                                    split=config.COCO_TRAIN_SPLIT,
                                    transform=train_tf, max_samples=max_train)
    val_ds = COCODetectionDataset(root_dir=root, split=config.COCO_VAL_SPLIT,
                                  transform=val_tf, max_samples=max_val)
    return train_ds, val_ds


def build_detection_loaders(root=config.COCO_ROOT, synthetic=False,
                            batch_size=config.BATCH_SIZE,
                            num_workers=config.NUM_WORKERS, device=None,
                            augment=True, max_train=None, max_val=None):
    train_ds, val_ds = build_detection_datasets(
        root=root, synthetic=synthetic, augment=augment,
        max_train=max_train, max_val=max_val)

    kwargs = _loader_kwargs(num_workers, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=detection_collate, **kwargs)
    return train_loader, val_loader


def num_classes_of(dataset):
    """Class count for a detection dataset, falling back to the COCO default."""
    return getattr(dataset, 'num_classes', config.NUM_CLASSES)


def class_names_of(dataset):
    names = getattr(dataset, 'dense_to_name', None)
    if names:
        return names
    return {i: str(i) for i in range(config.NUM_CLASSES)}


def images_only(loader):
    """Yield (images, dummy) pairs so calibration can share one profiling loop.

    Calibration only ever needs the input tensor, but the depth and detection
    loaders return different second elements. This adapter hides that.
    """
    for batch in loader:
        yield batch[0], torch.zeros(1)
