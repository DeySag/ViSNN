"""Matplotlib visualisation pipeline.

Turns raw output tensors into the artefacts a human can judge:

    * depth triptychs   RGB | ground truth | prediction, on a shared colour
                        scale so the two depth panels are directly comparable
    * detection overlays predicted boxes (solid) vs ground truth (dashed)
    * loss / metric curves

Matplotlib is forced onto the non-interactive Agg backend at import so this
works headless (Kaggle, Colab, CI) without a display.

The RGB tensor the model sees has ImageNet mean/std removed, so plotting it
directly gives a washed-out, wrongly-tinted image; `denormalize` puts it back.
"""

import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

import config
from data.transforms import denormalize
from losses.depth_loss import valid_mask

# Distinct, colour-blind-safe hues cycled per class id.
CLASS_COLOURS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2',
                 '#72B7B2', '#EECA3B', '#FF9DA6', '#9D755D', '#BAB0AC']


def _to_numpy_image(image_tensor):
    return denormalize(image_tensor.detach().cpu()).permute(1, 2, 0).numpy()


def _ensure_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Track A -- depth
# ---------------------------------------------------------------------------
def visualize_depth_batch(images, targets, predictions, save_path,
                          num_samples=config.VIS_SAMPLES, cmap='magma',
                          title=None):
    """Write an N-row RGB | ground-truth | prediction figure.

    Invalid ground-truth pixels (no LiDAR return) are masked to NaN and render
    as the colormap's 'bad' colour, so the sparsity of the target is visible
    rather than being drawn as "0 metres, very close".
    """
    images = images.detach().cpu()
    targets = targets.detach().cpu()
    predictions = predictions.detach().cpu()

    n = min(num_samples, images.shape[0])
    if n == 0:
        return None

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(color='#202020')

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n), squeeze=False)

    for row in range(n):
        target = targets[row, 0]
        prediction = predictions[row, 0]
        mask = valid_mask(target)

        # Shared colour scale, driven by the valid ground truth. Letting each
        # panel autoscale makes a bad prediction look plausible.
        if mask.any():
            vmin = float(target[mask].min())
            vmax = float(target[mask].max())
        else:
            vmin, vmax = 0.0, config.DEPTH_MAX
        if vmax <= vmin:
            vmax = vmin + 1e-3

        target_vis = target.clone()
        target_vis[~mask] = float('nan')

        axes[row][0].imshow(_to_numpy_image(images[row]))
        axes[row][0].set_title('RGB input' if row == 0 else '')

        axes[row][1].imshow(np.ma.masked_invalid(target_vis.numpy()),
                            cmap=colormap, vmin=vmin, vmax=vmax)
        axes[row][1].set_title('Ground truth (LiDAR)' if row == 0 else '')

        im = axes[row][2].imshow(prediction.numpy(), cmap=colormap,
                                 vmin=vmin, vmax=vmax)
        axes[row][2].set_title('SNN prediction' if row == 0 else '')

        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

        cbar = fig.colorbar(im, ax=axes[row][2], fraction=0.046, pad=0.02)
        cbar.set_label('depth (m)', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path


@torch.no_grad()
def visualize_depth_model(model, loader, device, save_path,
                          num_samples=config.VIS_SAMPLES, timesteps=1,
                          title=None):
    """Pull one validation batch, predict, and write the triptych figure."""
    from models.snn import spiking_forward

    was_training = model.training
    model.eval()
    try:
        images, targets = next(iter(loader))
    except StopIteration:
        model.train(was_training)
        return None

    images_dev = images.to(device)
    predictions = spiking_forward(model, images_dev, timesteps=timesteps).cpu()
    model.train(was_training)

    return visualize_depth_batch(images, targets, predictions, save_path,
                                 num_samples=num_samples, title=title)


# ---------------------------------------------------------------------------
# Track B -- detection
# ---------------------------------------------------------------------------
def _draw_boxes(ax, boxes, labels, scores, size, class_names, dashed=False):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) * size for v in boxes[i])
        label = int(labels[i])
        colour = CLASS_COLOURS[label % len(CLASS_COLOURS)]

        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor=colour,
            facecolor='none', linestyle='--' if dashed else '-'))

        name = class_names.get(label, str(label)) if class_names else str(label)
        text = name if scores is None else f'{name} {float(scores[i]):.2f}'
        ax.text(x1, max(y1 - 3, 6), text, fontsize=7, color='white',
                bbox=dict(facecolor=colour, alpha=0.85, pad=1.2,
                          edgecolor='none'))


def visualize_detection_batch(images, detections, gt_boxes, gt_labels,
                              save_path, num_samples=config.VIS_SAMPLES,
                              class_names=None, title=None):
    """Predicted boxes (solid, scored) over ground truth (dashed)."""
    images = images.detach().cpu()
    n = min(num_samples, images.shape[0])
    if n == 0:
        return None

    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows),
                             squeeze=False)
    size = images.shape[-1]

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n:
            ax.axis('off')
            continue

        ax.imshow(_to_numpy_image(images[idx]))
        _draw_boxes(ax, gt_boxes[idx], gt_labels[idx], None, size,
                    class_names, dashed=True)
        det = detections[idx]
        _draw_boxes(ax, det['boxes'].cpu(), det['labels'].cpu(),
                    det['scores'].cpu(), size, class_names, dashed=False)
        ax.set_title(f'{len(det["boxes"])} detection(s)  |  '
                     f'{len(gt_boxes[idx])} ground truth', fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path


@torch.no_grad()
def visualize_detection_model(model, loader, device, save_path,
                              num_samples=config.VIS_SAMPLES,
                              class_names=None, score_threshold=0.2,
                              title=None):
    was_training = model.training
    model.eval()
    try:
        images, gt_boxes, gt_labels = next(iter(loader))
    except StopIteration:
        model.train(was_training)
        return None

    detections = model.detect(images.to(device),
                              score_threshold=score_threshold)
    model.train(was_training)

    return visualize_detection_batch(images, detections, gt_boxes, gt_labels,
                                     save_path, num_samples=num_samples,
                                     class_names=class_names, title=title)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
def plot_curves(history, save_path, title='Training history'):
    """Plot a {series_name: [values]} history.

    Loss series share the left axis; metric series (mAP, delta1, ...) get a
    twinned right axis so a 0..1 accuracy is not flattened by a large loss.
    """
    if not history:
        return None

    loss_keys = [k for k in history if 'loss' in k.lower() or 'rmse' in k.lower()]
    metric_keys = [k for k in history if k not in loss_keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    for key in loss_keys:
        values = history[key]
        ax.plot(range(1, len(values) + 1), values, marker='o', label=key)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss / error')
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if metric_keys:
        ax2 = ax.twinx()
        for key in metric_keys:
            values = history[key]
            ax2.plot(range(1, len(values) + 1), values, marker='s',
                     linestyle='--', label=key)
        ax2.set_ylabel('metric')
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, loc='best', fontsize=9)
    ax.set_title(title)
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_spike_rates(report, save_path, title='Spike rate per layer'):
    """Bar chart of per-layer firing rates -- the energy story in one picture."""
    layers = report.get('layers', {})
    if not layers:
        return None

    names = list(layers.keys())
    values = [layers[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.32), 4.5))
    ax.bar(range(len(names)), values, color='#4C78A8')
    ax.axhline(report.get('overall', 0.0), color='#E45756', linestyle='--',
               label=f"overall {report.get('overall', 0.0):.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel('fraction of neurons firing')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
