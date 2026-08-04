"""SSD MultiBox loss: smooth-L1 localisation plus cross-entropy confidence.

Two details make SSD actually train:

Matching. The heads emit ~1194 predictions per image while an image holds only a
handful of objects. Regression targets exist only for priors matched to a
ground-truth box, so the localisation loss is computed over positives only.
Averaging over all priors drowns the real targets in ~1190 meaningless ones.

Hard-negative mining. With ~99% of priors labelled background, a plain mean
cross-entropy is dominated by easy negatives and the model converges to
"predict background everywhere". Only the hardest negatives per image
contribute.

Both terms are normalised by the number of positives, the standard SSD
convention, keeping the two terms on a comparable scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.box_utils import match_priors


class MultiBoxLoss(nn.Module):
    """Composite SSD objective: localisation + confidence.

    Args:
        priors: [P, 4] prior boxes in cxcywh, normalized. Registered as a
            buffer so it follows the module across `.to(device)`.
    """

    def __init__(self, priors, num_classes=config.NUM_CLASSES,
                 iou_threshold=config.SSD_IOU_THRESHOLD,
                 neg_pos_ratio=config.SSD_NEG_POS_RATIO,
                 variances=config.SSD_LOC_VARIANCES):
        super().__init__()
        self.register_buffer('priors', priors.clone().detach(),
                             persistent=False)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
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

        # --- Confidence: positives + hard negatives -------------------------
        flat_scores = pred_scores.view(-1, self.num_classes)
        flat_labels = label_targets.view(-1)
        per_prior_loss = F.cross_entropy(
            flat_scores, flat_labels, reduction='none').view(batch_size, -1)

        # Rank negatives by loss within each image. Positives are pushed to the
        # bottom so they cannot be selected as negatives too.
        negative_loss = per_prior_loss.clone()
        negative_loss[positives] = -1.0
        _, loss_rank = negative_loss.sort(dim=1, descending=True)
        _, rank = loss_rank.sort(dim=1)

        pos_per_image = positives.sum(dim=1, keepdim=True)
        num_negatives = torch.clamp(
            self.neg_pos_ratio * pos_per_image,
            max=num_priors - 1,
        )
        negatives = rank < num_negatives                    # [B, P]

        selected = positives | negatives
        cls_loss = per_prior_loss[selected].sum()

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
