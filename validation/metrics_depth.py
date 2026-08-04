"""Depth metrics.

The standard KITTI/Eigen suite. Every metric is computed over valid LiDAR
pixels only, and accumulated over the whole validation set before the final
division -- averaging per-batch RMSE values would weight a batch with 40 valid
pixels the same as one with 40,000.

Reported quantities:
    RMSE       root-mean-square error, metres (lower better)
    RMSE_log   RMSE in log space, penalises near-field error more
    MAE        mean absolute error, metres
    AbsRel     mean |d - d*| / d*
    SqRel      mean (d - d*)^2 / d*
    delta1/2/3 fraction of pixels within 1.25 / 1.25^2 / 1.25^3 of truth
"""

import torch

import config
from losses.depth_loss import valid_mask

EPS = 1e-6


class DepthMetrics:
    """Streaming accumulator for the depth metric suite."""

    def __init__(self, min_depth=config.DEPTH_MIN, max_depth=config.DEPTH_MAX):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self.n = 0.0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.sum_sq_log = 0.0
        self.sum_abs_rel = 0.0
        self.sum_sq_rel = 0.0
        self.delta = [0.0, 0.0, 0.0]

    @torch.no_grad()
    def update(self, predicted, target):
        mask = valid_mask(target, self.min_depth, self.max_depth)
        if not mask.any():
            return

        pred = predicted[mask].float().clamp(self.min_depth, self.max_depth)
        gt = target[mask].float()

        diff = pred - gt
        self.n += gt.numel()
        self.sum_sq += (diff ** 2).sum().item()
        self.sum_abs += diff.abs().sum().item()
        self.sum_abs_rel += (diff.abs() / gt.clamp(min=EPS)).sum().item()
        self.sum_sq_rel += ((diff ** 2) / gt.clamp(min=EPS)).sum().item()

        log_diff = torch.log(pred.clamp(min=EPS)) - torch.log(gt.clamp(min=EPS))
        self.sum_sq_log += (log_diff ** 2).sum().item()

        ratio = torch.max(pred / gt.clamp(min=EPS), gt / pred.clamp(min=EPS))
        for i, power in enumerate((1, 2, 3)):
            self.delta[i] += (ratio < 1.25 ** power).float().sum().item()

    def compute(self):
        if self.n == 0:
            return {k: float('nan') for k in
                    ('rmse', 'rmse_log', 'mae', 'abs_rel', 'sq_rel',
                     'delta1', 'delta2', 'delta3', 'num_pixels')}
        return {
            'rmse': (self.sum_sq / self.n) ** 0.5,
            'rmse_log': (self.sum_sq_log / self.n) ** 0.5,
            'mae': self.sum_abs / self.n,
            'abs_rel': self.sum_abs_rel / self.n,
            'sq_rel': self.sum_sq_rel / self.n,
            'delta1': self.delta[0] / self.n,
            'delta2': self.delta[1] / self.n,
            'delta3': self.delta[2] / self.n,
            'num_pixels': self.n,
        }

    def __str__(self):
        m = self.compute()
        return (f"RMSE {m['rmse']:.4f} m | MAE {m['mae']:.4f} m | "
                f"AbsRel {m['abs_rel']:.4f} | RMSE_log {m['rmse_log']:.4f} | "
                f"d1 {m['delta1']:.4f} | d2 {m['delta2']:.4f} | "
                f"d3 {m['delta3']:.4f}")


@torch.no_grad()
def evaluate_depth(model, loader, device, max_batches=None, timesteps=1,
                   verbose=False):
    """Run the validation loop and return the metric dict."""
    from models.snn import spiking_forward

    was_training = model.training
    model.eval()
    metrics = DepthMetrics()

    for i, (images, depths) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        predicted = spiking_forward(model, images, timesteps=timesteps)
        metrics.update(predicted, depths)
        if verbose and (i + 1) % 10 == 0:
            print(f'  eval batch {i + 1}: {metrics}')

    model.train(was_training)
    return metrics.compute()


@torch.no_grad()
def evaluate_depth_rmse(model, loader, device, max_batches=None):
    """Scalar RMSE -- the objective signature `search_lambda` expects."""
    return evaluate_depth(model, loader, device, max_batches)['rmse']


def format_metrics(metrics, prefix=''):
    order = ['rmse', 'mae', 'abs_rel', 'sq_rel', 'rmse_log',
             'delta1', 'delta2', 'delta3']
    parts = [f'{k}={metrics[k]:.4f}' for k in order if k in metrics]
    return f"{prefix}{'  '.join(parts)}"
