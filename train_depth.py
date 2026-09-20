"""Track A entrypoint: FastDepth-style depth estimation on a spiking backbone.

    python train_depth.py --synthetic --epochs 2          # no dataset needed
    python train_depth.py --data-root datasets/kitti --epochs 10

Pipeline: build loaders -> calibrate the continuous backbone -> convert to
StrictT1SFN -> freeze -> train the continuous decoder -> validate, visualise,
checkpoint.

Passing ``--run-dir results/<tag>`` redirects every artifact of the run into
that directory and additionally writes per-epoch ``metrics.csv``, a
``config.json`` snapshot and an energy report inside ``summary.json`` --
the layout the ablation runner expects.
"""

import argparse
import functools
import json
import os
import time

import torch
import torch.optim as optim

import config
import utils
from calibration.energy import build_energy_report, format_energy_report
from calibration.lambda_search import measure_spike_rate, search_lambda
from data.loaders import build_depth_loaders
from losses.depth_loss import DepthLoss
from models.snn import reset_spiking_state, set_spike_tracking, spiking_forward
from pipeline import build_depth_pipeline
from validation.metrics_depth import evaluate_depth, evaluate_depth_rmse, format_metrics
from validation.visualize import plot_curves, plot_spike_rates, visualize_depth_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train the spiking depth model')
    parser.add_argument('--data-root', default=config.KITTI_ROOT)
    parser.add_argument('--synthetic', action='store_true',
                        help='use generated KITTI-shaped data (no download)')
    parser.add_argument('--backbone', default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet50'])
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--spatial-mode', default=config.KITTI_SPATIAL_MODE,
                        choices=['crop', 'resize'])
    parser.add_argument('--output-activation', default='relu',
                        choices=['relu', 'softplus', 'linear'])
    parser.add_argument('--no-convert', action='store_true',
                        help='continuous control: skip calibration/conversion '
                             'and train the head on the frozen continuous trunk')

    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--alpha', type=float, default=config.GRAD_LOSS_ALPHA,
                        help='weight of the spatial-gradient loss term')
    parser.add_argument('--gradient-mode', default='matching',
                        choices=['smoothness', 'matching'])
    parser.add_argument('--loss-type', default='silog',
                        choices=['silog', 'rmse'],
                        help='base loss: silog (Eigen KITTI standard) or rmse (legacy)')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)

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
    parser.add_argument('--eigen-crop', action='store_true',
                        help='restrict depth metrics to the Eigen evaluation '
                             'crop (comparable to published KITTI numbers)')

    parser.add_argument('--limit-train', type=int, default=None)
    parser.add_argument('--limit-val', type=int, default=None)
    parser.add_argument('--eval-batches', type=int, default=config.MAX_EVAL_BATCHES,
                        help='validation cap; -1 also means uncapped')
    parser.add_argument('--full-eval', action='store_true',
                        help='evaluate the entire validation split (overrides '
                             '--eval-batches)')
    parser.add_argument('--run-dir', default=None,
                        help='directory for this run\'s artifacts; also enables '
                             'metrics.csv / config.json / energy reporting')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--tag', default='depth')
    return parser.parse_args()


def resolve_eval_batches(args):
    """--full-eval wins; negative --eval-batches counts as uncapped too."""
    if args.full_eval:
        return None
    if args.eval_batches is None or args.eval_batches < 0:
        return None
    return args.eval_batches


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    timesteps=1, grad_clip=config.GRAD_CLIP,
                    log_interval=config.LOG_INTERVAL):
    """One pass over the training set. Only the decoder receives gradients."""
    model.train()          # the encoder is pinned to eval() by the override
    loss_meter = utils.AverageMeter()
    rmse_meter = utils.AverageMeter()
    start = time.perf_counter()

    for step, (images, gt_depths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        gt_depths = gt_depths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Step 1: frozen spiking encoder. no_grad is not strictly required
        # (its parameters have requires_grad=False) but it avoids retaining
        # activations for a backward pass that will never reach them.
        with torch.no_grad():
            features = spiking_forward(model.encoder, images,
                                       timesteps=timesteps)

        # Step 2: continuous decoder -- this is where the graph starts.
        predicted = model.decode(features)

        # Step 3: masked loss + spatial gradient term.
        loss, parts = criterion(predicted, gt_depths, return_parts=True)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)

        # Step 4: update the decoder only.
        optimizer.step()

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        # Log actual RMSE for monitoring regardless of training loss type
        from losses.depth_loss import masked_rmse, valid_mask
        with torch.no_grad():
            train_rmse = masked_rmse(predicted, gt_depths, valid_mask(gt_depths)).item()
        rmse_meter.update(train_rmse, batch_size)

        if log_interval and (step + 1) % log_interval == 0:
            base_key = 'silog' if args.loss_type == 'silog' else 'rmse'
            base_val = parts.get(base_key, parts.get('rmse', torch.tensor(float('nan'))))
            print(f'  epoch {epoch} [{step + 1}/{len(loader)}]  '
                  f'loss {loss_meter.avg:.4f}  {base_key} {base_val.item():.4f}  '
                  f'train_rmse {rmse_meter.avg:.4f} m  '
                  f'valid {parts["valid_fraction"].item():.3f}')

    return {'loss': loss_meter.avg, 'train_rmse': rmse_meter.avg,
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
    print('== Track A: spiking depth estimation ==')
    print(f'device: {device} | torch {torch.__version__}')

    # ---- Data ------------------------------------------------------------
    train_loader, val_loader = build_depth_loaders(
        root=args.data_root, synthetic=args.synthetic,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=device, spatial_mode=args.spatial_mode,
        max_train=args.limit_train, max_val=args.limit_val)
    print(f'[data] train {len(train_loader.dataset)} sample(s) | '
          f'val {len(val_loader.dataset)} sample(s) | '
          f'batch {args.batch_size}')

    # ---- Calibration and conversion --------------------------------------
    threshold_path = (os.path.join(config.CHECKPOINT_DIR,
                                   f'{args.tag}_thresholds.npz')
                      if not args.no_convert else None)
    model, _thresholds = build_depth_pipeline(
        train_loader, device, backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        output_activation=args.output_activation,
        calibration_batches=min(args.calibration_batches, len(train_loader)),
        percentile=args.percentile, crop_margin=args.crop_margin,
        lambda_=args.lambda_, fire_fn=args.fire_fn, timesteps=args.timesteps,
        n_levels=args.n_levels, threshold_path=threshold_path,
        convert=not args.no_convert)

    if args.search_lambda:
        print('[calibrate] searching lambda...')
        objective = evaluate_depth_rmse
        if args.eigen_crop:
            objective = functools.partial(evaluate_depth_rmse,
                                          eigen_crop=True)
        best_lambda, _scores = search_lambda(
            model, val_loader, objective,
            grid=config.LAMBDA_SEARCH_GRID, device=device, mode='min',
            max_batches=eval_batches)
        args.lambda_ = best_lambda

    # ---- Training --------------------------------------------------------
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    criterion = DepthLoss(alpha=args.alpha, gradient_mode=args.gradient_mode,
                          loss_type=args.loss_type)

    csv_writer = None
    if run_dir:
        csv_writer = utils.MetricsCSVWriter(
            os.path.join(run_dir, 'metrics.csv'),
            ['epoch', 'loss', 'train_rmse', 'val_rmse', 'val_abs_rel',
             'val_delta1', 'seconds', 'best'])

    history = {'loss': [], 'train_rmse': [], 'val_rmse': [], 'delta1': []}
    best_rmse = float('inf')
    best_epoch = 0
    final_metrics = None

    try:
        for epoch in range(1, args.epochs + 1):
            stats = train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, epoch, timesteps=args.timesteps)
            scheduler.step()

            metrics = evaluate_depth(model, val_loader, device,
                                     max_batches=eval_batches,
                                     timesteps=args.timesteps,
                                     eigen_crop=args.eigen_crop)

            history['loss'].append(stats['loss'])
            history['train_rmse'].append(stats['train_rmse'])
            history['val_rmse'].append(metrics['rmse'])
            history['delta1'].append(metrics['delta1'])
            final_metrics = metrics

            print(f'epoch {epoch}/{args.epochs}  '
                  f'loss {stats["loss"]:.4f}  ({stats["seconds"]:.1f}s)')
            print(format_metrics(metrics, prefix='  val: '))

            is_best = metrics['rmse'] < best_rmse
            if csv_writer:
                csv_writer.append({
                    'epoch': epoch,
                    'loss': f'{stats["loss"]:.6f}',
                    'train_rmse': f'{stats["train_rmse"]:.6f}',
                    'val_rmse': f'{metrics["rmse"]:.6f}',
                    'val_abs_rel': f'{metrics.get("abs_rel", float("nan")):.6f}',
                    'val_delta1': f'{metrics["delta1"]:.6f}',
                    'seconds': f'{stats["seconds"]:.3f}',
                    'best': int(is_best),
                })

            visualize_depth_model(
                model, val_loader, device, out_path(
                    f'{args.tag}_epoch{epoch:02d}.png'),
                timesteps=args.timesteps,
                title=f'Epoch {epoch} -- val RMSE {metrics["rmse"]:.3f} m')

            if is_best:
                best_rmse = metrics['rmse']
                best_epoch = epoch
                utils.save_checkpoint({
                    'epoch': epoch,
                    'decoder_state': model.head.state_dict(),
                    'metrics': metrics,
                    'args': vars(args),
                }, f'{args.tag}_best.pt')
                print(f'  new best RMSE {best_rmse:.4f} m -> checkpoint saved')
    finally:
        if csv_writer:
            csv_writer.close()

    # ---- Reporting -------------------------------------------------------
    plot_curves(history, out_path(f'{args.tag}_curves.png'),
                title='Track A -- spiking depth estimation')

    reset_spiking_state(model.encoder)
    spike = measure_spike_rate(model.encoder, val_loader, device, max_batches=3)
    plot_spike_rates(spike, out_path(f'{args.tag}_spike_rates.png'))
    set_spike_tracking(model.encoder, enabled=False)

    summary = {
        'track': 'depth',
        'best_val_rmse': best_rmse,
        'best_epoch': best_epoch,
        'overall_spike_rate': spike['overall'],
        'history': history,
        'args': vars(args),
    }

    # Full metric suite from the final epoch (aggregate.py reads these fields;
    # None when the run trained for zero epochs).
    summary['final_metrics'] = final_metrics

    energy = build_energy_report(model.encoder, val_loader, device,
                                 input_size=config.INPUT_SIZE,
                                 max_batches=3, timesteps=args.timesteps,
                                 batch_size=args.batch_size)
    summary['energy'] = energy

    summary_path = out_path('summary.json' if run_dir
                            else f'{args.tag}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(f'\nbest val RMSE: {best_rmse:.4f} m (epoch {best_epoch})')
    print(f'overall spike rate: {spike["overall"]:.4f}')
    print(format_energy_report(energy))
    print(f'artifacts -> {os.path.dirname(summary_path)}')


if __name__ == '__main__':
    main()
