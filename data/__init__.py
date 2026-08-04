"""Data pipeline: datasets, transforms, loaders, and synthetic stand-ins."""

from data.coco import COCODetectionDataset, detection_collate
from data.kitti import KITTIDepthDataset, build_index, depth_collate
from data.loaders import (
    build_depth_datasets,
    build_depth_loaders,
    build_detection_datasets,
    build_detection_loaders,
    class_names_of,
    images_only,
    num_classes_of,
)
from data.synthetic import SyntheticDepthDataset, SyntheticDetectionDataset
from data.transforms import (
    DepthJointTransform,
    DetectionTransform,
    align_depth_target,
    denormalize,
    load_rgb,
)

__all__ = [
    'COCODetectionDataset',
    'DepthJointTransform',
    'DetectionTransform',
    'KITTIDepthDataset',
    'SyntheticDepthDataset',
    'SyntheticDetectionDataset',
    'align_depth_target',
    'build_depth_datasets',
    'build_depth_loaders',
    'build_detection_datasets',
    'build_detection_loaders',
    'build_index',
    'class_names_of',
    'denormalize',
    'depth_collate',
    'detection_collate',
    'images_only',
    'load_rgb',
    'num_classes_of',
]
