import torch
import torch.nn.functional as F


def calculate_rmse(pred, target):
    """RMSE metric (float)."""
    return torch.sqrt(F.mse_loss(pred, target)).item()


def compute_depth_loss(predicted, target, alpha=0.1):
    """Track A: RMSE + spatial-gradient smoothness term."""
    rmse = torch.sqrt(F.mse_loss(predicted, target))
    grad_x = torch.mean(torch.abs(predicted[:, :, :, :-1] - predicted[:, :, :, 1:]))
    grad_y = torch.mean(torch.abs(predicted[:, :, :-1, :] - predicted[:, :, 1:, :]))
    return rmse + alpha * (grad_x + grad_y)


def compute_multibox_loss(pred_locs, pred_scores, gt_locs, gt_labels):
    """Track B: SSD MultiBox loss (Smooth-L1 for boxes + CE for classes)."""
    loc_loss = F.smooth_l1_loss(pred_locs, gt_locs, reduction='mean')
    class_loss = F.cross_entropy(pred_scores, gt_labels, reduction='mean')
    return loc_loss + class_loss