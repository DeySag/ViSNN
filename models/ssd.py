"""SSD detector on a spiking MobileNetV2 trunk.

Layout at 224x224 input:

    feature map   size    channels   aspect ratios      priors/loc   priors
    -------------------------------------------------------------------------
    stage1        14x14      96      [2]                    4         784
    stage2         7x7     1280      [2, 3]                 6         294
    extra1         4x4      512      [2, 3]                 6          96
    extra2         2x2      256      [2]                    4          16
    extra3         1x1      256      [2]                    4           4
                                                             total   1194

The MobileNet stages are converted to spikes and frozen; the extra layers and
both prediction heads remain continuous and trainable.
"""

import math

import torch
import torch.nn as nn

import config
from models.backbone import MobileNetV2SSDBackbone
from models.box_utils import batched_nms, decode

# Anchor scales, coarse-to-fine. The final entry is the s_{m+1} used for the
# geometric-mean prior on the last map.
DEFAULT_SCALES = (0.10, 0.30, 0.50, 0.70, 0.90, 1.05)
DEFAULT_ASPECT_RATIOS = ((2,), (2, 3), (2, 3), (2,), (2,))


class PriorBox(nn.Module):
    """Generates the fixed prior (anchor) boxes in cxcywh, normalized 0..1."""

    def __init__(self, feature_sizes, scales=DEFAULT_SCALES,
                 aspect_ratios=DEFAULT_ASPECT_RATIOS, clip=True):
        super().__init__()
        if len(scales) != len(feature_sizes) + 1:
            raise ValueError(
                f'Need len(scales) == len(feature_sizes) + 1; got '
                f'{len(scales)} and {len(feature_sizes)}')
        if len(aspect_ratios) != len(feature_sizes):
            raise ValueError('aspect_ratios must have one entry per feature map')

        self.feature_sizes = tuple(feature_sizes)
        self.scales = tuple(scales)
        self.aspect_ratios = tuple(tuple(a) for a in aspect_ratios)
        self.clip = clip
        self.register_buffer('priors', self._generate(), persistent=False)

    @property
    def num_priors_per_location(self):
        # 1 square at s_k, 1 square at sqrt(s_k * s_k+1), then 2 per ratio.
        return tuple(2 + 2 * len(ar) for ar in self.aspect_ratios)

    def _generate(self):
        boxes = []
        for k, f_k in enumerate(self.feature_sizes):
            s_k = self.scales[k]
            s_k_prime = math.sqrt(s_k * self.scales[k + 1])
            for i in range(f_k):
                for j in range(f_k):
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k
                    boxes.append([cx, cy, s_k, s_k])
                    boxes.append([cx, cy, s_k_prime, s_k_prime])
                    for ratio in self.aspect_ratios[k]:
                        root = math.sqrt(ratio)
                        boxes.append([cx, cy, s_k * root, s_k / root])
                        boxes.append([cx, cy, s_k / root, s_k * root])

        priors = torch.tensor(boxes, dtype=torch.float32)
        if self.clip:
            # Keep centres and extents inside the frame; boxes stay in cxcywh so
            # only the sizes need bounding.
            priors[:, 2:].clamp_(min=1e-4, max=1.0)
        return priors

    def forward(self):
        return self.priors


class SSDHeads(nn.Module):
    """Per-feature-map localisation and classification convolutions.

    Outputs are concatenated across maps into
        loc:    [B, P, 4]
        scores: [B, P, num_classes]
    with prior order matching `PriorBox` exactly (map -> row -> col -> anchor).
    """

    def __init__(self, feature_channels, priors_per_location,
                 num_classes=config.NUM_CLASSES):
        super().__init__()
        if len(feature_channels) != len(priors_per_location):
            raise ValueError('feature_channels / priors_per_location mismatch')

        self.num_classes = num_classes
        self.loc_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()

        for channels, n_priors in zip(feature_channels, priors_per_location):
            self.loc_heads.append(
                nn.Conv2d(channels, n_priors * 4, kernel_size=3, padding=1))
            self.cls_heads.append(
                nn.Conv2d(channels, n_priors * num_classes,
                          kernel_size=3, padding=1))

        self._init_weights()

    def _init_weights(self):
        for head in list(self.loc_heads) + list(self.cls_heads):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        # Bias the classifier towards background. Without this, ~1194 priors
        # start at uniform class probability, the background term dominates the
        # first steps, and the loss spikes before it settles.
        prior_background = 0.99
        bias_value = -math.log((1 - prior_background) / prior_background)
        for head in self.cls_heads:
            with torch.no_grad():
                bias = head.bias.view(-1, self.num_classes)
                bias[:, 0] = bias_value

    def forward(self, features):
        locs, scores = [], []
        for feature, loc_head, cls_head in zip(features, self.loc_heads,
                                               self.cls_heads):
            batch = feature.shape[0]
            # [B, A*4, H, W] -> [B, H, W, A*4] -> [B, H*W*A, 4]; the permute
            # must come before the reshape or the anchor ordering silently
            # stops matching PriorBox.
            loc = loc_head(feature).permute(0, 2, 3, 1).contiguous()
            locs.append(loc.view(batch, -1, 4))

            score = cls_head(feature).permute(0, 2, 3, 1).contiguous()
            scores.append(score.view(batch, -1, self.num_classes))

        return torch.cat(locs, dim=1), torch.cat(scores, dim=1)


class SpikingSSD(nn.Module):
    """MobileNetV2 (spiking, frozen) + extra layers + SSD heads (continuous)."""

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True,
                 backbone=None, freeze_trunk=True):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_trunk = freeze_trunk
        self.backbone = backbone or MobileNetV2SSDBackbone(pretrained=pretrained)

        feature_sizes = self.backbone.feature_sizes
        feature_channels = self.backbone.feature_channels

        self.priorbox = PriorBox(feature_sizes)
        self.heads = SSDHeads(feature_channels,
                              self.priorbox.num_priors_per_location,
                              num_classes=num_classes)

    @property
    def priors(self):
        return self.priorbox.priors

    @property
    def trainable_modules(self):
        """Everything the optimizer should touch: extras + both heads."""
        return nn.ModuleList([
            self.backbone.extra1, self.backbone.extra2, self.backbone.extra3,
            self.heads,
        ])

    def trainable_parameters(self):
        return self.trainable_modules.parameters()

    def train(self, mode=True):
        """Train the extras and heads; keep the spiking trunk in eval mode.

        Same reasoning as the depth decoder: the frozen MobileNet stages must
        keep using their calibrated BatchNorm running statistics.
        """
        super().train(mode)
        if self.freeze_trunk:
            self.backbone.stage1.eval()
            self.backbone.stage2.eval()
        return self

    def forward(self, x):
        features = self.backbone(x)
        loc, scores = self.heads(features)
        return loc, scores

    @torch.no_grad()
    def detect(self, x, score_threshold=config.SSD_SCORE_THRESHOLD,
               nms_threshold=config.SSD_NMS_THRESHOLD, top_k=config.SSD_TOP_K):
        """Full inference: forward -> decode -> per-class NMS.

        Returns one dict per image with 'boxes' [N, 4] xyxy normalized,
        'scores' [N] and 'labels' [N] (dense class ids, never 0).
        """
        loc, scores = self.forward(x)
        return decode_detections(loc, scores, self.priors.to(loc.device),
                                 score_threshold=score_threshold,
                                 nms_threshold=nms_threshold, top_k=top_k)


def decode_detections(loc, scores, priors,
                      score_threshold=config.SSD_SCORE_THRESHOLD,
                      nms_threshold=config.SSD_NMS_THRESHOLD,
                      top_k=config.SSD_TOP_K,
                      variances=config.SSD_LOC_VARIANCES):
    """Turn raw SSD outputs into per-image detection lists."""
    batch_size, _, num_classes = scores.shape
    probs = torch.softmax(scores, dim=-1)
    results = []

    for b in range(batch_size):
        boxes = decode(loc[b], priors, variances).clamp(0.0, 1.0)
        image_probs = probs[b]

        # Class 0 is background and is never emitted as a detection.
        best_scores, best_labels = image_probs[:, 1:].max(dim=1)
        best_labels = best_labels + 1

        keep_mask = best_scores > score_threshold
        if not keep_mask.any():
            results.append({
                'boxes': boxes.new_zeros((0, 4)),
                'scores': boxes.new_zeros((0,)),
                'labels': torch.zeros(0, dtype=torch.int64, device=boxes.device),
            })
            continue

        candidate_boxes = boxes[keep_mask]
        candidate_scores = best_scores[keep_mask]
        candidate_labels = best_labels[keep_mask]

        # Degenerate boxes survive decode when the head is untrained; drop them
        # so NMS is not comparing zero-area rectangles.
        valid = ((candidate_boxes[:, 2] > candidate_boxes[:, 0]) &
                 (candidate_boxes[:, 3] > candidate_boxes[:, 1]))
        candidate_boxes = candidate_boxes[valid]
        candidate_scores = candidate_scores[valid]
        candidate_labels = candidate_labels[valid]

        keep = batched_nms(candidate_boxes, candidate_scores,
                           candidate_labels, nms_threshold, top_k)
        results.append({
            'boxes': candidate_boxes[keep],
            'scores': candidate_scores[keep],
            'labels': candidate_labels[keep],
        })

    return results


def build_ssd_model(num_classes=config.NUM_CLASSES, pretrained=True,
                    device=None):
    model = SpikingSSD(num_classes=num_classes, pretrained=pretrained)
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def sanity_check_ssd(model, input_size=config.INPUT_SIZE, device='cpu'):
    """Verify head output length equals the prior count."""
    was_training = model.training
    model.eval()
    dummy = torch.zeros(2, 3, input_size, input_size, device=device)
    loc, scores = model(dummy)
    num_priors = model.priors.shape[0]

    if loc.shape[1] != num_priors or scores.shape[1] != num_priors:
        raise RuntimeError(
            f'Head/prior mismatch: loc {loc.shape[1]}, scores '
            f'{scores.shape[1]}, priors {num_priors}. The head reshape order '
            'and the PriorBox generation order have diverged.')
    model.train(was_training)
    return {'num_priors': num_priors, 'loc': tuple(loc.shape),
            'scores': tuple(scores.shape)}
