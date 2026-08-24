"""Track B entrypoint: SSD object detection on a spiking MobileNet trunk.

    python train_ssd.py --synthetic --epochs 3            # no dataset needed
    python train_ssd.py --data-root datasets/coco --epochs 10

The MobileNet stages are quantized into binary spikes and frozen; the SSD extra
layers and the localisation/classification heads stay continuous and are the
only things trained.

Passing ``--run-dir results/<tag>`` redirects every artifact of the run into
that directory and additionally writes per-epoch ``metrics.csv``, a
``config.json`` snapshot and an energy report inside ``summary.json`` --
the layout the ablation runner expects.
"""

import argparse
import json
import os
import time

import torch
import torch.optim as optim

import config
import utils
from calibration.energy import build_energy_report, format_energy_report
from calibration.lambda_search import measure_spike_rate, search_lambda
from data.loaders import build_detection_loaders, class_names_of, num_classes_of
from losses.multibox_loss import MultiBoxLoss
from models.snn import reset_spiking_state, set_spike_tracking
from pipeline import build_ssd_pipeline
from validation.metrics_detection import evaluate_detection, evaluate_detection_map
from validation.visualize import (
    plot_curves,
    plot_spike_rates,
    visualize_detection_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the spiking SSD model')
    parser.add_argument('--data-root', default=config.COCO_ROOT)
    parser.add_argument('--synthetic', action='store_true',
                        help='use generated COCO-shaped data (no download)')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--no-convert', action='store_true',
                        help='continuous control: keep the MobileNet stages '
                             'continuous and train heads/extras on them')

    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--neg-pos-ratio', type=int,
                        default=config.SSD_NEG_POS_RATIO)
    parser.add_argument('--iou-threshold', type=float,
                        default=config.SSD_IOU_THRESHOLD)

    parser.add_argument('--timesteps', type=int, default=config.TIMESTEPS)
    parser.add_argument('--fire-fn', default=config.FIRE_FN,
                        choices=['binary', 'mtn'])
    parser.add_argument('--lambda', dest='lambda_', type=float,
                        default=config.LAMBDA)
    parser.add_argument('--n-levels', type=int, default=config.N_LEVELS)
    parser.add_argument('--search-lambda', action='store_true')
    parser.add_argument('--calibration-batches', type=int,
                        default=config.CALIBRATION_BATCHES)
    parser.add_argument('--percentile', type=float, default=config.TOP_P)
    parser.add_argument('--crop-margin', type=int, default=config.CROP_MARGIN)

    parser.add_argument('--limit-train', type=int, default=None)
    parser.add_argument('--limit-val', type=int, default=None)
    parser.add_argument('--eval-batches', type=int, default=config.MAX_EVAL_BATCHES,
                        help='validation cap; -1 also means uncapped')
    parser.add_argument('--full-eval', action='store_true',
                        help='evaluate the entire validation split (overrides '
                             '--eval-batches)')
    parser.add_argument('--coco-metrics', action='store_true',
                        help='also report mAP@[.5:.95] (10x slower)')
    parser.add_argument('--run-dir', default=None,
                        help="directory for this run's artifacts; also enables "
                             'metrics.csv / config.json / energy reporting')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--tag', default='ssd')
    return parser.parse_args()


def resolve_eval_batches(args):
    """--full-eval wins; negative --eval-batches counts as uncapped too."""
    if args.full_eval:
        return None
    if args.eval_batches is None or args.eval_batches < 0:
        return None
    return args.eval_batches


def forward_split(model, images, timesteps=1):
    """Frozen spiking trunk under no_grad, then the continuous graph.

    Returns (loc, scores). The trunk features are produced without autograd,
    so the computational graph begins at the extra layers.
    """
    with torch.no_grad():
        reset_spiking_state(model.backbone)
        if timesteps <= 1:
            f1 = model.backbone.stage1(images)
            f2 = model.backbone.stage2(f1)
        else:
            acc1 = acc2 = None
            for _ in range(timesteps):
                step1 = model.backbone.stage1(images)
                step2 = model.backbone.stage2(step1)
                acc1 = step1 if acc1 is None else acc1 + step1
                acc2 = step2 if acc2 is None else acc2 + step2
            f1, f2 = acc1 / timesteps, acc2 / timesteps

    f3 = model.backbone.extra1(f2)
    f4 = model.backbone.extra2(f3)
    f5 = model.backbone.extra3(f4)
    return model.heads([f1, f2, f3, f4, f5])


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    timesteps=1, grad_clip=config.GRAD_CLIP,
                    log_interval=config.LOG_INTERVAL):
    model.train()          # the spiking trunk is pinned to eval() by override
    loss_meter = utils.AverageMeter()
    loc_meter = utils.AverageMeter()
    cls_meter = utils.AverageMeter()
    start = time.perf_counter()

    for step, (images, gt_boxes, gt_labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loc, scores = forward_split(model, images, timesteps=timesteps)
        loss, parts = criterion(loc, scores, gt_boxes, gt_labels,
                                return_parts=True)

        if parts['num_positives'] == 0:
            continue  # nothing matched anywhere in this batch

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        loc_meter.update(parts['loc'].item(), batch_size)
        cls_meter.update(parts['cls'].item(), batch_size)

        if log_interval and (step + 1) % log_interval == 0:
            print(f'  epoch {epoch} [{step + 1}/{len(loader)}]  '
                  f'loss {loss_meter.avg:.4f}  loc {loc_meter.avg:.4f}  '
                  f'cls {cls_meter.avg:.4f}  pos {parts["num_positives"]}')

    return {'loss': loss_meter.avg, 'loc_loss': loc_meter.avg,
            'cls_loss': cls_meter.avg,
            'seconds': time.perf_counter() - start}


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    config.ensure_dirs()
    eval_batches = resolve_eval_batches(args)

    run_dir = args.run_dir
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'config.json'), 'w',
                  encoding='utf-8') as handle:
            json.dump(vars(args), handle, indent=2)

    def out_path(name):
        return os.path.join(run_dir or config.RESULTS_DIR, name)

    device = utils.get_device(args.device)
    print('== Track B: spiking SSD object detection ==')
    print(f'device: {device} | torch {torch.__version__}')

    # ---- Data ------------------------------------------------------------
    train_loader, val_loader = build_detection_loaders(
        root=args.data_root, synthetic=args.synthetic,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=device, max_train=args.limit_train, max_val=args.limit_val)

    num_classes = num_classes_of(train_loader.dataset)
    class_names = class_names_of(train_loader.dataset)
    print(f'[data] train {len(train_loader.dataset)} sample(s) | '
          f'val {len(val_loader.dataset)} sample(s) | '
          f'{num_classes - 1} foreground class(es)')

    # ---- Calibration and conversion --------------------------------------
    threshold_path = (os.path.join(config.CHECKPOINT_DIR,
                                   f'{args.tag}_thresholds.npz')
                      if not args.no_convert else None)
    model, _thresholds = build_ssd_pipeline(
        train_loader, device, num_classes=num_classes,
        pretrained=not args.no_pretrained,
        calibration_batches=min(args.calibration_batches, len(train_loader)),
        percentile=args.percentile, crop_margin=args.crop_margin,
        lambda_=args.lambda_, fire_fn=args.fire_fn, timesteps=args.timesteps,
        n_levels=args.n_levels, threshold_path=threshold_path,
        convert=not args.no_convert)

    if args.search_lambda:
        print('[calibrate] searching lambda...')
        best_lambda, _scores = search_lambda(
            model, val_loader, evaluate_detection_map,
            grid=config.LAMBDA_SEARCH_GRID, device=device, mode='max',
            max_batches=eval_batches)
        args.lambda_ = best_lambda

    # ---- Training --------------------------------------------------------
    optimizer = optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    criterion = MultiBoxLoss(model.priors, num_classes=num_classes,
                             iou_threshold=args.iou_threshold,
                             neg_pos_ratio=args.neg_pos_ratio).to(device)

    csv_writer = None
    if run_dir:
        csv_writer = utils.MetricsCSVWriter(
            os.path.join(run_dir, 'metrics.csv'),
            ['epoch', 'loss', 'loc_loss', 'cls_loss', 'mAP',
             'mAP@[.5:.95]', 'num_classes_evaluated', 'seconds', 'best'])

    history = {'loss': [], 'loc_loss': [], 'cls_loss': [], 'mAP': []}
    best_map = -1.0
    best_epoch = 0
    final_metrics = None

    try:
        for epoch in range(1, args.epochs + 1):
            stats = train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, epoch, timesteps=args.timesteps)
            scheduler.step()

            metrics = evaluate_detection(model, val_loader, device,
                                         max_batches=eval_batches,
                                         num_classes=num_classes,
                                         class_names=class_names,
                                         coco_style=args.coco_metrics)
            final_metrics = metrics

            history['loss'].append(stats['loss'])
            history['loc_loss'].append(stats['loc_loss'])
            history['cls_loss'].append(stats['cls_loss'])
            history['mAP'].append(metrics['mAP'])

            print(f'epoch {epoch}/{args.epochs}  loss {stats["loss"]:.4f} '
                  f'(loc {stats["loc_loss"]:.4f} / cls {stats["cls_loss"]:.4f})  '
                  f'({stats["seconds"]:.1f}s)')
            print(f'  val: mAP@0.5 {metrics["mAP"]:.4f} over '
                  f'{metrics["num_classes_evaluated"]} class(es)')
            if args.coco_metrics:
                print(f'       mAP@[.5:.95] {metrics["mAP@[.5:.95]"]:.4f}')

            is_best = metrics['mAP'] > best_map
            if csv_writer:
                csv_writer.append({
                    'epoch': epoch,
                    'loss': f'{stats["loss"]:.6f}',
                    'loc_loss': f'{stats["loc_loss"]:.6f}',
                    'cls_loss': f'{stats["cls_loss"]:.6f}',
                    'mAP': f'{metrics["mAP"]:.6f}',
                    'mAP@[.5:.95]': (
                        f'{metrics["mAP@[.5:.95]"]:.6f}'
                        if 'mAP@[.5:.95]' in metrics else ''),
                    'num_classes_evaluated':
                        metrics.get('num_classes_evaluated', ''),
                    'seconds': f'{stats["seconds"]:.3f}',
                    'best': int(is_best),
                })

            visualize_detection_model(
                model, val_loader, device, out_path(
                    f'{args.tag}_epoch{epoch:02d}.png'),
                class_names=class_names,
                title=f'Epoch {epoch} -- mAP@0.5 {metrics["mAP"]:.3f}')

            if is_best:
                best_map = metrics['mAP']
                best_epoch = epoch
                utils.save_checkpoint({
                    'epoch': epoch,
                    'heads_state': model.heads.state_dict(),
                    'extras_state': {
                        'extra1': model.backbone.extra1.state_dict(),
                        'extra2': model.backbone.extra2.state_dict(),
                        'extra3': model.backbone.extra3.state_dict(),
                    },
                    'metrics': {k: v for k, v in metrics.items()
                                if k != 'per_class'},
                    'args': vars(args),
                }, f'{args.tag}_best.pt')
                print(f'  new best mAP {best_map:.4f} -> checkpoint saved')
    finally:
        if csv_writer:
            csv_writer.close()

    # ---- Reporting -------------------------------------------------------
    plot_curves(history, out_path(f'{args.tag}_curves.png'),
                title='Track B -- spiking SSD detection')

    reset_spiking_state(model.backbone)
    spike = measure_spike_rate(model, val_loader, device, max_batches=3)
    plot_spike_rates(spike, out_path(f'{args.tag}_spike_rates.png'))
    set_spike_tracking(model, enabled=False)

    summary = {
        'track': 'ssd',
        'best_val_mAP': best_map,
        'best_epoch': best_epoch,
        'overall_spike_rate': spike['overall'],
        'history': history,
        'args': vars(args),
        # Full metric suite from the final epoch (aggregate.py reads these).
        'final_metrics': ({k: v for k, v in final_metrics.items()
                           if k != 'per_class'} if final_metrics else None),
    }

    energy = build_energy_report(model.backbone, val_loader, device,
                                 input_size=config.INPUT_SIZE,
                                 max_batches=3, timesteps=args.timesteps,
                                 batch_size=args.batch_size)
    summary['energy'] = energy

    summary_path = out_path('summary.json' if run_dir
                            else f'{args.tag}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(f'\nbest val mAP@0.5: {best_map:.4f} (epoch {best_epoch})')
    print(f'overall spike rate: {spike["overall"]:.4f}')
    print(format_energy_report(energy))
    print(f'artifacts -> {os.path.dirname(summary_path)}')


if __name__ == '__main__':
    main()
