"""COCO detection dataset.

The annotation JSON is parsed directly with the standard library; `pycocotools`
is not a dependency (it requires a C toolchain, awkward to install on Windows).
Only images[], annotations[] and categories[] are needed from the file.

Two coordinate conventions meet here:
    * the file stores boxes as absolute [x, y, width, height];
    * SSD wants normalized [x_min, y_min, x_max, y_max] in [0, 1].
The conversion happens once, at parse time, in `_parse_annotations`.

COCO category ids are sparse (1..90 with gaps). They are remapped to a dense
1..80, leaving index 0 free for the background class the MultiBox loss needs.
"""

import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset

import config
from data.transforms import DetectionTransform, load_rgb


class COCODetectionDataset(Dataset):
    """Streams (image, boxes, labels) triples from a COCO instances split.

    Returns:
        image:  float32 [3, 224, 224], ImageNet-normalized.
        boxes:  float32 [N, 4], normalized xyxy in [0, 1].
        labels: int64  [N], dense class ids in 1..80 (0 = background).

    N varies per image, so use `detection_collate` with the DataLoader.
    """

    def __init__(self, root_dir=config.COCO_ROOT, split=config.COCO_TRAIN_SPLIT,
                 transform=None, annotation_file=None, image_dir=None,
                 max_samples=None, drop_empty=True, min_box_size=1e-3):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or DetectionTransform(
            augment=split.startswith('train'))
        self.min_box_size = min_box_size

        self.annotation_file = annotation_file or os.path.join(
            root_dir, 'annotations', f'instances_{split}.json')
        self.image_dir = image_dir or os.path.join(root_dir, split)

        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(
                f'COCO annotations not found: {self.annotation_file}\n'
                'Download the COCO 2017 train/val images + the '
                '`annotations_trainval2017.zip` bundle and point '
                'config.COCO_ROOT at the extracted folder, or run with '
                '--synthetic to exercise the pipeline without data.'
            )

        self._parse_annotations(drop_empty, max_samples)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_annotations(self, drop_empty, max_samples):
        with open(self.annotation_file, 'r', encoding='utf-8') as handle:
            raw = json.load(handle)

        # Sparse COCO ids (1..90, with gaps) -> dense 1..80.
        categories = sorted(raw['categories'], key=lambda c: c['id'])
        self.coco_id_to_dense = {c['id']: i + 1
                                 for i, c in enumerate(categories)}
        self.dense_to_name = {i + 1: c['name']
                              for i, c in enumerate(categories)}
        self.dense_to_name[0] = '__background__'
        self.num_classes = len(categories) + 1

        images_by_id = {img['id']: img for img in raw['images']}

        grouped = defaultdict(list)
        for ann in raw['annotations']:
            if ann.get('iscrowd', 0):
                continue  # crowd regions have no usable single box
            image_id = ann['image_id']
            meta = images_by_id.get(image_id)
            if meta is None:
                continue

            width, height = float(meta['width']), float(meta['height'])
            x, y, w, h = (float(v) for v in ann['bbox'])
            if w <= 0 or h <= 0:
                continue

            # [x, y, w, h] absolute -> [x1, y1, x2, y2] normalized, clipped to
            # the frame (a few COCO boxes overhang the image bounds).
            x1 = min(max(x / width, 0.0), 1.0)
            y1 = min(max(y / height, 0.0), 1.0)
            x2 = min(max((x + w) / width, 0.0), 1.0)
            y2 = min(max((y + h) / height, 0.0), 1.0)
            if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                continue

            label = self.coco_id_to_dense.get(ann['category_id'])
            if label is None:
                continue
            grouped[image_id].append(([x1, y1, x2, y2], label))

        self.samples = []
        for image_id in sorted(images_by_id):
            entries = grouped.get(image_id, [])
            if drop_empty and not entries:
                continue  # SSD gets no learning signal from an all-negative image
            meta = images_by_id[image_id]
            path = os.path.join(self.image_dir, meta['file_name'])
            self.samples.append({
                'image_id': image_id,
                'path': path,
                'boxes': [e[0] for e in entries],
                'labels': [e[1] for e in entries],
            })
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(
                f'No usable samples parsed from {self.annotation_file}.')

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = load_rgb(sample['path'])

        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return self.transform(image, boxes, labels)


def detection_collate(batch):
    """Stack images; keep boxes/labels as lists because N varies per image."""
    images = torch.stack([item[0] for item in batch], dim=0)
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels
