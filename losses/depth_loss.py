"""Depth loss: masked RMSE plus a spatial-gradient term.

Three details that a naive formulation gets wrong, all handled here:

1. Invalid pixels. KITTI ground truth is sparse LiDAR; a pixel with no return
   is stored as 0.0. Averaging over those zeros trains the network to predict
   0 metres over ~85% of the frame, so every term is masked by
   `target > DEPTH_MIN`.

2. The square root at zero. d/dx sqrt(x) is unbounded at x = 0, so as the MSE
   approaches zero the gradient blows up and the run NaNs out. An epsilon inside
   the square root bounds it.

3. The squaring. `(predicted - target) * 2` is a doubling, not a square: it
   makes the "RMSE" the square root of a signed mean, which becomes NaN the
   moment the model over-predicts on average.

On the gradient term: taking gradients of the prediction alone is a smoothness
prior that blurs edges; matching the prediction's gradients to the target's
sharpen boundaries. `mode='smoothness'` is the smoothing formulation (the
goal prior), `mode='matching'` compares against ground-truth gradients.
"""

import torch

import config

EPS = 1e-6


def valid_mask(target, min_depth=config.DEPTH_MIN, max_depth=config.DEPTH_MAX):
    """True where the LiDAR ground truth carries a usable measurement."""
    return (target > min_depth) & (target <= max_depth) & torch.isfinite(target)


def masked_rmse(predicted, target, mask=None, eps=EPS):
    """Root-mean-square error over valid pixels only."""
    if mask is None:
        mask = valid_mask(target)
    count = mask.sum()
    if count == 0:
        # No LiDAR returns in this batch: contribute nothing, but keep the
        # result attached to the graph so `.backward()` still has a path.
        return predicted.sum() * 0.0
    squared = ((predicted - target) ** 2)[mask]
    return torch.sqrt(squared.sum() / count + eps)


def masked_l1(predicted, target, mask=None):
    if mask is None:
        mask = valid_mask(target)
    count = mask.sum()
    if count == 0:
        return predicted.sum() * 0.0
    return (predicted - target).abs()[mask].sum() / count


def image_gradients(tensor):
    """Forward differences along x and y. Shapes shrink by 1 on that axis."""
    grad_x = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
    grad_y = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
    return grad_x, grad_y


def gradient_loss(predicted, target=None, mode='smoothness', mask=None):
    """Spatial-gradient term.

    mode='smoothness': mean |grad(pred)|. Penalises all structure, giving
        smoother, more regularised depth maps.
    mode='matching': mean |grad(pred) - grad(gt)| over pixel pairs where both
        ground-truth samples are valid. Rewards edges present in the ground
        truth, producing genuinely sharper boundaries.
    """
    grad_x, grad_y = image_gradients(predicted)

    if mode == 'smoothness':
        return grad_x.abs().mean() + grad_y.abs().mean()

    if mode == 'matching':
        if target is None:
            raise ValueError("mode='matching' requires a target")
        if mask is None:
            mask = valid_mask(target)
        # A difference is only meaningful when both of its two source pixels
        # are valid, so the mask is ANDed with its own shift.
        mask_x = mask[:, :, :, :-1] & mask[:, :, :, 1:]
        mask_y = mask[:, :, :-1, :] & mask[:, :, 1:, :]
        tgt_x, tgt_y = image_gradients(target)

        total = predicted.sum() * 0.0
        if mask_x.any():
            total = total + (grad_x - tgt_x).abs()[mask_x].mean()
        if mask_y.any():
            total = total + (grad_y - tgt_y).abs()[mask_y].mean()
        return total

    raise ValueError(f'Unknown gradient mode: {mode!r}')


def compute_depth_loss(predicted, target, alpha=config.GRAD_LOSS_ALPHA,
                       gradient_mode='smoothness', return_parts=False):
    """L = RMSE_masked(pred, gt) + alpha * gradient_term."""
    mask = valid_mask(target)
    rmse = masked_rmse(predicted, target, mask)
    grad = gradient_loss(predicted, target, mode=gradient_mode, mask=mask)
    total = rmse + alpha * grad

    if return_parts:
        return total, {'rmse': rmse.detach(), 'gradient': grad.detach(),
                       'valid_fraction': mask.float().mean().detach()}
    return total


class DepthLoss(torch.nn.Module):
    """`compute_depth_loss` as a module, for use inside an nn pipeline."""

    def __init__(self, alpha=config.GRAD_LOSS_ALPHA,
                 gradient_mode='smoothness'):
        super().__init__()
        self.alpha = alpha
        self.gradient_mode = gradient_mode

    def forward(self, predicted, target, return_parts=False):
        return compute_depth_loss(predicted, target, alpha=self.alpha,
                                  gradient_mode=self.gradient_mode,
                                  return_parts=return_parts)


def scale_invariant_log_loss(predicted, target, lambda_=0.85, mask=None):
    """SILog (Eigen et al.) -- optional alternative objective.

    Insensitive to a global scale factor, which suits monocular depth where
    absolute scale is ambiguous. Not used by default.
    """
    if mask is None:
        mask = valid_mask(target)
    count = mask.sum()
    if count == 0:
        return predicted.sum() * 0.0

    log_diff = (torch.log(predicted.clamp(min=EPS)) -
                torch.log(target.clamp(min=EPS)))[mask]
    return torch.sqrt((log_diff ** 2).mean() -
                      lambda_ * (log_diff.mean() ** 2) + EPS)


def berhu_loss(predicted, target, mask=None):
    """Reverse-Huber: L1 for small residuals, L2 for large. FastDepth's choice."""
    if mask is None:
        mask = valid_mask(target)
    if mask.sum() == 0:
        return predicted.sum() * 0.0

    diff = (predicted - target).abs()[mask]
    threshold = 0.2 * diff.max().detach()
    if threshold <= 0:
        return diff.mean()
    l2_part = (diff ** 2 + threshold ** 2) / (2 * threshold)
    return torch.where(diff <= threshold, diff, l2_part).mean()


__all__ = ['DepthLoss', 'berhu_loss', 'compute_depth_loss', 'gradient_loss',
           'image_gradients', 'masked_l1', 'masked_rmse',
           'scale_invariant_log_loss', 'valid_mask']
