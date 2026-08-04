"""Track A entrypoint: FastDepth-style depth estimation on a spiking backbone.

    python train_depth.py --synthetic --epochs 2          # no dataset needed
    python train_depth.py --data-root datasets/kitti --epochs 10

Pipeline: build loaders -> calibrate the continuous backbone -> convert to
StrictT1SFN -> freeze -> train the continuous decoder -> validate, visualise,
checkpoint.
"""

import argparse
import json
import os
import time

import torch
import torch.optim as optim

import config
import utils
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

    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY)
    parser.add_argument('--alpha', type=float, default=config.GRAD_LOSS_ALPHA,
                        help='weight of the spatial-gradient loss term')
    parser.add_argument('--gradient-mode', default='smoothness',
                        choices=['smoothness', 'matching'])
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

    parser.add_argument('--limit-train', type=int, default=None)
    parser.add_argument('--limit-val', type=int, default=None)
    parser.add_argument('--eval-batches', type=int, default=config.MAX_EVAL_BATCHES)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--tag', default='depth')
    return parser.parse_args()


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

        # Step 3: masked RMSE + spatial gradient term.
        loss, parts = criterion(predicted, gt_depths, return_parts=True)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)

        # Step 4: update the decoder only.
        optimizer.step()

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        rmse_meter.update(parts['rmse'].item(), batch_size)

        if log_interval and (step + 1) % log_interval == 0:
            print(f'  epoch {epoch} [{step + 1}/{len(loader)}]  '
                  f'loss {loss_meter.avg:.4f}  rmse {rmse_meter.avg:.4f} m  '
                  f'valid {parts["valid_fraction"].item():.3f}')

    return {'loss': loss_meter.avg, 'train_rmse': rmse_meter.avg,
            'seconds': time.perf_counter() - start}


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    config.ensure_dirs()

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
    threshold_path = os.path.join(config.CHECKPOINT_DIR,
                                  f'{args.tag}_thresholds.npz')
    model, _thresholds = build_depth_pipeline(
        train_loader, device, backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        output_activation=args.output_activation,
        calibration_batches=min(args.calibration_batches, len(train_loader)),
        percentile=args.percentile, crop_margin=args.crop_margin,
        lambda_=args.lambda_, fire_fn=args.fire_fn, timesteps=args.timesteps,
        n_levels=args.n_levels, threshold_path=threshold_path)

    if args.search_lambda:
        print('[calibrate] searching lambda...')
        best_lambda, _scores = search_lambda(
            model, val_loader, evaluate_depth_rmse,
            grid=config.LAMBDA_SEARCH_GRID, device=device, mode='min',
            max_batches=args.eval_batches)
        args.lambda_ = best_lambda

    # ---- Training --------------------------------------------------------
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    criterion = DepthLoss(alpha=args.alpha, gradient_mode=args.gradient_mode)

    history = {'loss': [], 'train_rmse': [], 'val_rmse': [], 'delta1': []}
    best_rmse = float('inf')

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, criterion,
                                device, epoch, timesteps=args.timesteps)
        scheduler.step()

        metrics = evaluate_depth(model, val_loader, device,
                                 max_batches=args.eval_batches,
                                 timesteps=args.timesteps)

        history['loss'].append(stats['loss'])
        history['train_rmse'].append(stats['train_rmse'])
        history['val_rmse'].append(metrics['rmse'])
        history['delta1'].append(metrics['delta1'])

        print(f'epoch {epoch}/{args.epochs}  '
              f'loss {stats["loss"]:.4f}  ({stats["seconds"]:.1f}s)')
        print(format_metrics(metrics, prefix='  val: '))

        visualize_depth_model(
            model, val_loader, device,
            os.path.join(config.RESULTS_DIR,
                         f'{args.tag}_epoch{epoch:02d}.png'),
            timesteps=args.timesteps,
            title=f'Epoch {epoch} -- val RMSE {metrics["rmse"]:.3f} m')

        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            utils.save_checkpoint({
                'epoch': epoch,
                'decoder_state': model.head.state_dict(),
                'metrics': metrics,
                'args': vars(args),
            }, f'{args.tag}_best.pt')
            print(f'  new best RMSE {best_rmse:.4f} m -> checkpoint saved')

    # ---- Reporting -------------------------------------------------------
    plot_curves(history, os.path.join(config.RESULTS_DIR,
                                      f'{args.tag}_curves.png'),
                title='Track A -- spiking depth estimation')

    reset_spiking_state(model.encoder)
    spike = measure_spike_rate(model.encoder, val_loader, device, max_batches=3)
    plot_spike_rates(spike, os.path.join(config.RESULTS_DIR,
                                         f'{args.tag}_spike_rates.png'))
    set_spike_tracking(model.encoder, enabled=False)

    summary = {
        'best_val_rmse': best_rmse,
        'overall_spike_rate': spike['overall'],
        'history': history,
        'args': vars(args),
    }
    summary_path = os.path.join(config.RESULTS_DIR, f'{args.tag}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(f'\nbest val RMSE: {best_rmse:.4f} m')
    print(f'overall spike rate: {spike["overall"]:.4f}')
    print(f'artifacts -> {config.RESULTS_DIR}')


if __name__ == '__main__':
    main()
