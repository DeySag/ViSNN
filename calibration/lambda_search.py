"""Global scaling-factor (lambda) search.

Per-channel thresholds set the shape of the spiking response; lambda scales all
of them together. Lowering lambda lowers every effective threshold, so more
neurons fire and more information survives the T=1 quantization, at the cost of
a higher spike rate and therefore higher energy.

The search is a plain grid: evaluate a small held-out set at each lambda and
keep the best. It runs under `torch.no_grad()` and mutates only `lambda_`, so
it is cheap and leaves no state behind beyond the winning value.
"""

import torch

import config
from models.snn import set_lambda, set_spike_tracking, spike_report


@torch.no_grad()
def search_lambda(model, loader, evaluate_fn, grid=config.LAMBDA_SEARCH_GRID,
                  device=None, mode='min', max_batches=None, verbose=True,
                  restore_best=True):
    """Grid-search lambda against an arbitrary scalar objective.

    Args:
        model: a converted (spiking) model.
        loader: validation loader.
        evaluate_fn: `(model, loader, device, max_batches) -> float`.
        mode: 'min' for error-like scores (RMSE, loss), 'max' for mAP.

    Returns:
        (best_lambda, {lambda: score}).
    """
    if mode not in ('min', 'max'):
        raise ValueError("mode must be 'min' or 'max'")

    was_training = model.training
    model.eval()

    scores = {}
    for lambda_ in grid:
        set_lambda(model, lambda_)
        set_spike_tracking(model, enabled=True, reset=True)

        score = float(evaluate_fn(model, loader, device, max_batches))
        rate = spike_report(model)['overall']
        scores[lambda_] = score

        set_spike_tracking(model, enabled=False, reset=True)
        if verbose:
            print(f'  lambda={lambda_:<5.2f} score={score:.4f}  '
                  f'spike_rate={rate:.3f}')

    picker = min if mode == 'min' else max
    best_lambda = picker(scores, key=scores.get)

    set_lambda(model, best_lambda if restore_best else config.LAMBDA)
    model.train(was_training)

    if verbose:
        print(f'  -> best lambda = {best_lambda} '
              f'({scores[best_lambda]:.4f})')
    return best_lambda, scores


@torch.no_grad()
def measure_spike_rate(model, loader, device, max_batches=5):
    """Overall firing rate across the spiking layers, for energy reporting."""
    was_training = model.training
    model.eval()
    set_spike_tracking(model, enabled=True, reset=True)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        model(batch[0].to(device, non_blocking=True))

    report = spike_report(model)
    set_spike_tracking(model, enabled=False, reset=True)
    model.train(was_training)
    return report
