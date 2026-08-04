"""Bounding-box utilities for SSD.

All operations work in normalized coordinates (0..1 relative to the image),
which keeps the maths independent of the 224x224 input size.

Two representations are used:
    xyxy   = [x_min, y_min, x_max, y_max]   -- ground truth, IoU, NMS, output
    cxcywh = [c_x, c_y, w, h]               -- priors and network regression

`encode`/`decode` convert between a ground-truth box and the offsets the
network regresses, scaled by SSD's variance constants.
"""

import torch

import config


# ---------------------------------------------------------------------------
# Representation conversion
# ---------------------------------------------------------------------------
def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------
def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
           (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def jaccard(boxes_a, boxes_b):
    """IoU matrix between [N, 4] and [M, 4] xyxy boxes -> [N, M]."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]

    union = box_area(boxes_a)[:, None] + box_area(boxes_b)[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------
def encode(matched_xyxy, priors_cxcywh, variances=config.SSD_LOC_VARIANCES):
    """Ground-truth boxes -> regression targets, relative to their priors."""
    matched = xyxy_to_cxcywh(matched_xyxy)
    prior_cx, prior_cy, prior_w, prior_h = priors_cxcywh.unbind(-1)
    gt_cx, gt_cy, gt_w, gt_h = matched.unbind(-1)

    prior_w = prior_w.clamp(min=1e-9)
    prior_h = prior_h.clamp(min=1e-9)

    g_cx = (gt_cx - prior_cx) / (variances[0] * prior_w)
    g_cy = (gt_cy - prior_cy) / (variances[0] * prior_h)
    g_w = torch.log((gt_w / prior_w).clamp(min=1e-9)) / variances[1]
    g_h = torch.log((gt_h / prior_h).clamp(min=1e-9)) / variances[1]
    return torch.stack([g_cx, g_cy, g_w, g_h], dim=-1)


def decode(loc, priors_cxcywh, variances=config.SSD_LOC_VARIANCES):
    """Network regression outputs -> absolute xyxy boxes. Inverse of encode."""
    prior_cx, prior_cy, prior_w, prior_h = priors_cxcywh.unbind(-1)
    l_cx, l_cy, l_w, l_h = loc.unbind(-1)

    cx = prior_cx + l_cx * variances[0] * prior_w
    cy = prior_cy + l_cy * variances[0] * prior_h
    # Clamp the exponent: an untrained head can emit large values and overflow
    # to inf, which would poison NMS with NaNs.
    w = prior_w * torch.exp((l_w * variances[1]).clamp(max=10.0))
    h = prior_h * torch.exp((l_h * variances[1]).clamp(max=10.0))

    return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=-1))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match_priors(gt_boxes, gt_labels, priors_cxcywh,
                 iou_threshold=config.SSD_IOU_THRESHOLD,
                 variances=config.SSD_LOC_VARIANCES):
    """Assign each prior a ground-truth box (or background).

    Two passes, both needed:
      1. Every prior takes its best-overlapping ground truth, if that overlap
         clears `iou_threshold`.
      2. Every ground truth force-claims its single best prior regardless of
         threshold, so a small object that no prior overlaps well still gets
         one positive to train against.

    Returns:
        loc_targets:   [P, 4] encoded offsets (meaningless where label == 0).
        label_targets: [P] int64, 0 = background.
    """
    num_priors = priors_cxcywh.shape[0]
    device = priors_cxcywh.device

    if gt_boxes.numel() == 0:
        return (torch.zeros(num_priors, 4, device=device),
                torch.zeros(num_priors, dtype=torch.int64, device=device))

    priors_xyxy = cxcywh_to_xyxy(priors_cxcywh)
    overlaps = jaccard(gt_boxes, priors_xyxy)              # [G, P]

    best_gt_iou, best_gt_idx = overlaps.max(dim=0)         # per prior  -> [P]
    best_prior_iou, best_prior_idx = overlaps.max(dim=1)   # per gt     -> [G]

    # Pass 2: pin each ground truth to its best prior and make that assignment
    # unbeatable so pass-1's argmax cannot steal it back.
    best_gt_idx[best_prior_idx] = torch.arange(gt_boxes.shape[0], device=device)
    best_gt_iou[best_prior_idx] = 1.0

    matched_boxes = gt_boxes[best_gt_idx]                  # [P, 4]
    labels = gt_labels[best_gt_idx].clone()                # [P]
    labels[best_gt_iou < iou_threshold] = 0                # background

    loc_targets = encode(matched_boxes, priors_cxcywh, variances)
    return loc_targets, labels


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------
def _greedy_nms(boxes, scores, iou_threshold, top_k):
    """Reference greedy NMS. Correct but O(n^2) in Python -- the fallback."""
    order = scores.argsort(descending=True)[:top_k]
    keep = []
    while order.numel() > 0:
        best = order[0]
        keep.append(best.item())
        if order.numel() == 1:
            break
        ious = jaccard(boxes[best].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def nms(boxes, scores, iou_threshold=config.SSD_NMS_THRESHOLD, top_k=200):
    """Greedy NMS on [N, 4] xyxy boxes. Returns kept indices, best first.

    Dispatches to torchvision's fused CUDA/C++ kernel when available -- the
    Python loop below is identical in behaviour but roughly two orders of
    magnitude slower, which dominates validation time at ~1200 priors/image.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)

    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes, scores, iou_threshold)[:top_k]
    except ImportError:
        return _greedy_nms(boxes, scores, iou_threshold, top_k)


def batched_nms(boxes, scores, class_ids, iou_threshold, top_k):
    """Class-aware NMS: boxes of different classes never suppress each other."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)

    try:
        from torchvision.ops import batched_nms as tv_batched_nms
        return tv_batched_nms(boxes, scores, class_ids, iou_threshold)[:top_k]
    except ImportError:
        # Offset each class into its own coordinate block so a single NMS pass
        # cannot compute a non-zero IoU across classes.
        offsets = class_ids.to(boxes.dtype) * (boxes.max() + 1.0)
        return _greedy_nms(boxes + offsets[:, None], scores, iou_threshold,
                           top_k)
