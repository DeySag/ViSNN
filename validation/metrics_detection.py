"""Detection metrics: mean Average Precision.

Implements mAP from first principles (no pycocotools). For each class:

  1. Pool every detection of that class across the whole validation set and
     sort by confidence, descending.
  2. Walk the sorted list. A detection is a true positive if it overlaps an
     as-yet-unmatched ground-truth box of the same class in the same image with
     IoU >= threshold; otherwise it is a false positive. The unmatched
     bookkeeping is what makes duplicate detections count against you.
  3. Cumulative TP/FP give a precision-recall curve; AP is the area under it,
     using all-point interpolation (the current VOC/COCO convention; the older
     11-point sampling under-reports).

mAP is the mean AP over classes that appear in the ground truth; classes with
no ground truth are undefined and skipped rather than scored as zero.

`mAP@[.5:.95]` averages the whole procedure over ten IoU thresholds and is the
headline COCO number.
"""

import numpy as np
import torch

import config
from models.box_utils import jaccard

COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


class DetectionEvaluator:
    """Accumulates predictions and ground truth, then computes mAP."""

    def __init__(self, num_classes=config.NUM_CLASSES, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or {}
        self.reset()

    def reset(self):
        self.predictions = []   # per image: dict(boxes, scores, labels)
        self.ground_truth = []  # per image: dict(boxes, labels)

    @torch.no_grad()
    def update(self, predictions, gt_boxes, gt_labels):
        """
        Args:
            predictions: list of dicts with 'boxes' [N,4] xyxy, 'scores' [N],
                'labels' [N] -- exactly what `SpikingSSD.detect` returns.
            gt_boxes / gt_labels: per-image lists from `detection_collate`.
        """
        for pred, boxes, labels in zip(predictions, gt_boxes, gt_labels):
            self.predictions.append({
                'boxes': pred['boxes'].detach().cpu(),
                'scores': pred['scores'].detach().cpu(),
                'labels': pred['labels'].detach().cpu(),
            })
            self.ground_truth.append({
                'boxes': boxes.detach().cpu(),
                'labels': labels.detach().cpu(),
            })

    # ------------------------------------------------------------------
    @staticmethod
    def _average_precision(recall, precision):
        """Area under the PR curve, all-point interpolated."""
        # Sentinel endpoints so the curve spans recall 0..1.
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        # Make precision monotonically non-increasing from the right.
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        change = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))

    def _class_ap(self, class_id, iou_threshold):
        """AP for one class, or None when the class has no ground truth."""
        # Ground truth for this class, indexed by image.
        gt_by_image, num_gt = {}, 0
        for image_idx, gt in enumerate(self.ground_truth):
            keep = gt['labels'] == class_id
            if keep.any():
                boxes = gt['boxes'][keep]
                gt_by_image[image_idx] = {
                    'boxes': boxes,
                    'matched': np.zeros(len(boxes), dtype=bool),
                }
                num_gt += len(boxes)

        if num_gt == 0:
            return None

        # Detections for this class, pooled and sorted by confidence.
        records = []
        for image_idx, pred in enumerate(self.predictions):
            keep = pred['labels'] == class_id
            if keep.any():
                for box, score in zip(pred['boxes'][keep], pred['scores'][keep]):
                    records.append((float(score), image_idx, box))

        if not records:
            return 0.0

        records.sort(key=lambda r: r[0], reverse=True)

        tp = np.zeros(len(records), dtype=np.float64)
        fp = np.zeros(len(records), dtype=np.float64)

        for i, (_score, image_idx, box) in enumerate(records):
            entry = gt_by_image.get(image_idx)
            if entry is None:
                fp[i] = 1.0
                continue

            ious = jaccard(box.unsqueeze(0), entry['boxes']).squeeze(0).numpy()
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and not entry['matched'][best]:
                tp[i] = 1.0
                entry['matched'][best] = True   # one GT box, one credit
            else:
                fp[i] = 1.0

        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cum_tp / num_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        return self._average_precision(recall, precision)

    # ------------------------------------------------------------------
    def compute(self, iou_threshold=0.5, per_class=False):
        aps = {}
        for class_id in range(1, self.num_classes):   # 0 is background
            ap = self._class_ap(class_id, iou_threshold)
            if ap is not None:
                aps[class_id] = ap

        mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
        result = {'mAP': mean_ap, 'num_classes_evaluated': len(aps),
                  'iou_threshold': iou_threshold}
        if per_class:
            result['per_class'] = {
                self.class_names.get(cid, str(cid)): ap
                for cid, ap in sorted(aps.items())
            }
        return result

    def compute_coco(self, thresholds=COCO_IOU_THRESHOLDS):
        """mAP@[.5:.95] plus the two headline single-threshold numbers."""
        per_threshold = {float(t): self.compute(iou_threshold=float(t))['mAP']
                         for t in thresholds}
        return {
            'mAP@[.5:.95]': float(np.mean(list(per_threshold.values()))),
            'mAP@0.5': per_threshold.get(0.5, 0.0),
            'mAP@0.75': per_threshold.get(0.75, 0.0),
            'per_threshold': per_threshold,
        }


@torch.no_grad()
def evaluate_detection(model, loader, device, max_batches=None,
                       num_classes=config.NUM_CLASSES, class_names=None,
                       score_threshold=config.SSD_SCORE_THRESHOLD,
                       coco_style=False, verbose=False):
    """Validation loop for Track B. Returns the metric dict."""
    was_training = model.training
    model.eval()

    evaluator = DetectionEvaluator(num_classes=num_classes,
                                   class_names=class_names)

    for i, (images, gt_boxes, gt_labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        detections = model.detect(images, score_threshold=score_threshold)
        evaluator.update(detections, gt_boxes, gt_labels)
        if verbose and (i + 1) % 10 == 0:
            print(f'  eval batch {i + 1}')

    model.train(was_training)

    result = evaluator.compute(iou_threshold=0.5, per_class=True)
    if coco_style:
        result.update(evaluator.compute_coco())
    return result


@torch.no_grad()
def evaluate_detection_map(model, loader, device, max_batches=None):
    """Scalar mAP@0.5 -- the objective signature `search_lambda` expects."""
    return evaluate_detection(model, loader, device, max_batches,
                              num_classes=model.num_classes)['mAP']
