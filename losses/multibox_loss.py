"""SSD MultiBox loss: smooth-L1 localisation plus Focal Loss confidence.

Two details make SSD actually train:

Matching. The heads emit ~1194 predictions per image while an image holds only a
handful of objects. Regression targets exist only for priors matched to a
ground-truth box, so the localisation loss is computed over positives only.
Averaging over all priors drowns the real targets in ~1190 meaningless ones.

Focal Loss (Lin et al., RetinaNet). With ~99% of priors labelled background,
standard cross-entropy is dominated by easy negatives. Focal Loss natively
down-weights well-classified examples (both easy background and easy positives)
via a modulating factor (1 - p_t)^gamma, eliminating the need for hard-negative
mining. This is the modern drop-in replacement for CE + mining.

Both terms are normalised by the number of positives, the standard SSD
convention, keeping the two terms on a comparable scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.box_utils import match_priors


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Focal Loss (Lin et al., RetinaNet).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: [N, C] raw scores (no softmax)
        targets: [N] class indices in 0..C-1
        alpha: weighting factor for rare classes (0.25 default)
        gamma: focusing parameter (2.0 default)
        reduction: 'sum' or 'mean'
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
    p_t = (probs * targets_one_hot).sum(dim=1)  # probability of true class
    log_p_t = (log_probs * targets_one_hot).sum(dim=1)
    alpha_t = targets_one_hot * alpha + (1 - targets_one_hot) * (1 - alpha)
    alpha_t = alpha_t.sum(dim=1)
    loss = -alpha_t * (1 - p_t).pow(gamma) * log_p_t
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


class MultiBoxLoss(nn.Module):
    """Composite SSD objective: localisation + Focal Loss confidence.

    Args:
        priors: [P, 4] prior boxes in cxcywh, normalized. Registered as a
            buffer so it follows the module across `.to(device)`.
        alpha: Focal Loss alpha (class weight for background vs foreground)
        gamma: Focal Loss gamma (focusing parameter)
    """

    def __init__(self, priors, num_classes=config.NUM_CLASSES,
                 iou_threshold=config.SSD_IOU_THRESHOLD,
                 focal_alpha=0.25, focal_gamma=2.0,
                 variances=config.SSD_LOC_VARIANCES):
        super().__init__()
        self.register_buffer('priors', priors.clone().detach(),
                             persistent=False)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.variances = variances

    # ------------------------------------------------------------------
    def build_targets(self, gt_boxes, gt_labels, device):
        """Match every image's ground truth onto the shared prior grid."""
        loc_targets, label_targets = [], []
        for boxes, labels in zip(gt_boxes, gt_labels):
            loc_t, label_t = match_priors(
                boxes.to(device), labels.to(device), self.priors,
                iou_threshold=self.iou_threshold, variances=self.variances)
            loc_targets.append(loc_t)
            label_targets.append(label_t)
        return torch.stack(loc_targets), torch.stack(label_targets)

    # ------------------------------------------------------------------
    def forward(self, pred_locs, pred_scores, gt_boxes, gt_labels,
                return_parts=False):
        """
        Args:
            pred_locs:   [B, P, 4] regression outputs.
            pred_scores: [B, P, num_classes] raw logits.
            gt_boxes:    list of B tensors [N_i, 4], normalized xyxy.
            gt_labels:   list of B tensors [N_i], dense ids in 1..C-1.
        """
        device = pred_locs.device
        batch_size, num_priors = pred_locs.shape[:2]

        if num_priors != self.priors.shape[0]:
            raise ValueError(
                f'Prediction has {num_priors} priors but the loss was built '
                f'with {self.priors.shape[0]}. The head and PriorBox '
                'configurations disagree.')

        loc_targets, label_targets = self.build_targets(
            gt_boxes, gt_labels, device)

        positives = label_targets > 0                       # [B, P]
        num_positives = positives.sum()

        if num_positives == 0:
            # Nothing matched anywhere in the batch. Still return a graph-
            # connected zero so the training step is a no-op rather than a
            # crash.
            zero = pred_locs.sum() * 0.0 + pred_scores.sum() * 0.0
            if return_parts:
                return zero, {'loc': zero.detach(), 'cls': zero.detach(),
                              'num_positives': 0}
            return zero

        # --- Localisation: positives only -----------------------------------
        loc_loss = F.smooth_l1_loss(
            pred_locs[positives], loc_targets[positives], reduction='sum')

        # --- Confidence: Focal Loss over ALL priors (no hard-negative mining) ---
        # Focal Loss natively down-weights easy background examples via the
        # (1 - p_t)^gamma term, so we don't need to mine hard negatives.
        flat_scores = pred_scores.view(-1, self.num_classes)
        flat_labels = label_targets.view(-1)
        cls_loss = focal_loss(flat_scores, flat_labels,
                              alpha=self.focal_alpha, gamma=self.focal_gamma,
                              reduction='sum')

        # --- Normalise ------------------------------------------------------
        denominator = num_positives.clamp(min=1).float()
        loc_loss = loc_loss / denominator
        cls_loss = cls_loss / denominator
        total = loc_loss + cls_loss

        if return_parts:
            return total, {
                'loc': loc_loss.detach(),
                'cls': cls_loss.detach(),
                'num_positives': int(num_positives.item()),
            }
        return total


def compute_multibox_loss(pred_locs, pred_scores, gt_locs, gt_labels,
                          priors=None, num_classes=config.NUM_CLASSES,
                          criterion=None):
    """Functional wrapper around `MultiBoxLoss`.

    Pass either a prepared `criterion` or the `priors` tensor to build one on
    the fly. `gt_locs` / `gt_labels` are the per-image lists produced by
    `detection_collate`.
    """
    if criterion is None:
        if priors is None:
            raise ValueError('Provide either `criterion` or `priors`')
        criterion = MultiBoxLoss(priors, num_classes=num_classes).to(
            pred_locs.device)
    return criterion(pred_locs, pred_scores, gt_locs, gt_labels)


__all__ = ['MultiBoxLoss', 'compute_multibox_loss']
