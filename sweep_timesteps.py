"""T-sweep harness: find the minimal timestep where the accuracy/energy
tradeoff is justified (SFN, per arXiv:2510.23383).

For each (T, fire_fn) configuration:
    1. Reuse the once-calibrated spatial-masked channel-wise thresholds.
    2. Build a fresh spiking encoder (frozen SFN backbone).
    3. Optionally reuse a per-fire-fn lambda found by feature-MSE search.
    4. Train the continuous decoder (few epochs), evaluate depth RMSE.
    5. Estimate ANN vs SNN energy and feature-level MSE.

Emits results/sweep_results.csv + an RMSE-vs-T / energy-vs-T figure.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
from tqdm import tqdm

import config
from calibration.lambda_search import compute_feature_mse, search_lambda
from calibration.profile import collect_profiles, compute_channel_thresholds
from data import make_dataset
from energy import estimate_energy
from losses.depth_loss import calculate_rmse
from models.backbone import get_mobilenetv2_backbone
from models.decoder import SimpleDepthDecoder
from models.snn import convert_to_snn
from models.spiking_encoder import SpikingEncoder
from train_depth import run_validation, train_decoder


@torch.no_grad()
def collect_continuous_features(golden_encoder, loader, device, num_batches):
    chunks = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        chunks.append(golden_encoder(images.to(device)).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def collect_snn_features(encoder, loader, device, num_batches):
    chunks = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        chunks.append(encoder(images.to(device)).cpu())
    return torch.cat(chunks, dim=0)


def run_sweep(args, device, train_loader, val_loader):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Golden continuous features (fidelity reference).
    golden = torchvision_models.mobilenet_v2(weights='DEFAULT').features.to(device)
    golden.eval()
    continuous_features = collect_continuous_features(golden, val_loader, device,
                                                      config.CALIBRATION_BATCHES)
    print(f'Cached continuous features: {tuple(continuous_features.shape)}')

    # 2. Calibrate thresholds once on a probe backbone.
    probe = get_mobilenetv2_backbone(device)
    profiles = collect_profiles(probe, val_loader, device, config.CALIBRATION_BATCHES)
    channel_thresholds = compute_channel_thresholds(profiles, config.TOP_P, config.CROP_MARGIN)
    del probe
    print(f'Calibrated {len(channel_thresholds)} layers once; reusing across sweep.')

    # 3. Lambda: search once per fire function at T=1, reuse for all T.
    lambda_per_fn = {}
    if args.search_lambda:
        for fire_fn in config.SWEEP_FIRE_FNS:
            enc = build_encoder(device, timesteps=1, fire_fn=fire_fn,
                                lambda_=1.0, channel_thresholds=channel_thresholds)
            enc.eval()
            best, _ = search_lambda(enc, golden, val_loader, device,
                                    grid=config.LAMBDA_SEARCH_GRID,
                                    num_batches=config.CALIBRATION_BATCHES)
            lambda_per_fn[fire_fn] = best
            print(f'[lambda] {fire_fn}: best lambda={best:.3f}')
    else:
        lambda_per_fn = {fn: args.lambda_ for fn in config.SWEEP_FIRE_FNS}

    rows = []
    for fire_fn in config.SWEEP_FIRE_FNS:
        lam = lambda_per_fn[fire_fn]
        for timesteps in config.SWEEP_TIMESTEPS:
            print(f'\n=== T={timesteps} | {fire_fn} | lambda={lam} ===')

            encoder = build_encoder(device, timesteps, fire_fn, lam, channel_thresholds)
            encoder.eval()

            snn_features = collect_snn_features(encoder, val_loader, device,
                                                config.CALIBRATION_BATCHES)
            feature_mse = compute_feature_mse(snn_features, continuous_features)

            model = SimpleDepthDecoder(encoder).to(device)
            model.eval()
            train_decoder(model, train_loader, device, epochs=args.epochs)
            rmse = run_validation(model, val_loader, device, config.MAX_BATCHES)

            sample_images, _ = next(iter(val_loader))
            ann_energy, snn_energy, ac_fraction, firing_rate = estimate_energy(
                model, sample_images.to(device))
            energy_ratio = snn_energy / max(ann_energy, 1e-9)

            print(f'T={timesteps} {fire_fn}: RMSE={rmse:.4f} | featMSE={feature_mse:.6f} | '
                  f'energy_ratio={energy_ratio:.3f} | firing_rate={firing_rate:.3f}')

            rows.append({
                'timesteps': timesteps,
                'fire_fn': fire_fn,
                'lambda': lam,
                'feature_mse': round(feature_mse, 6),
                'rmse': round(rmse, 4),
                'ann_energy_pJ': round(ann_energy, 1),
                'snn_energy_pJ': round(snn_energy, 1),
                'energy_ratio': round(energy_ratio, 4),
                'ac_fraction': round(ac_fraction, 4),
                'firing_rate': round(firing_rate, 4),
            })

    write_results(rows)
    plot_results(rows)


def build_encoder(device, timesteps, fire_fn, lambda_, channel_thresholds):
    backbone = get_mobilenetv2_backbone(device)
    convert_to_snn(backbone, channel_thresholds, device, lambda_=lambda_,
                   fire_fn=fire_fn, timesteps=timesteps, n_levels=config.N_LEVELS)
    return SpikingEncoder(backbone.features, timesteps=timesteps)


def write_results(rows):
    path = os.path.join(config.RESULTS_DIR, 'sweep_results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults written to {path}')


def plot_results(rows):
    path = os.path.join(config.RESULTS_DIR, 'sweep_plot.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for fire_fn in config.SWEEP_FIRE_FNS:
        sub = [r for r in rows if r['fire_fn'] == fire_fn]
        ts = [r['timesteps'] for r in sub]
        rmse = [r['rmse'] for r in sub]
        energy = [r['energy_ratio'] for r in sub]
        ax1.plot(ts, rmse, marker='o', label=fire_fn)
        ax2.plot(ts, energy, marker='o', label=fire_fn)

    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Timesteps T')
    ax1.set_ylabel('Depth RMSE')
    ax1.set_title('Accuracy vs Timesteps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Timesteps T')
    ax2.set_ylabel('SNN / ANN Energy Ratio')
    ax2.set_title('Energy vs Timesteps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'Plot saved to {path}')


def main(args):
    from train_depth import resolve_data_root, set_seed

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Workspace initialized. Using device: {device}')

    data_root = resolve_data_root(args.data_root)
    print(f'Dataset: {config.DATASET} | Data root: {data_root}')

    train_ds = make_dataset(config.DATASET, data_root, mode='train', input_size=config.INPUT_SIZE)
    val_ds = make_dataset(config.DATASET, data_root, mode='val', input_size=config.INPUT_SIZE)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)

    run_sweep(args, device, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sweep timesteps x fire functions to find the RMSE/energy tradeoff knee.')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root (overrides config DATA_ROOT/TARTANAIR_ROOT).')
    parser.add_argument('--epochs', type=int, default=config.SWEEP_EPOCHS,
                        help=f'Decoder epochs per configuration (default {config.SWEEP_EPOCHS}).')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=config.LAMBDA,
                        help=f'Fixed global scaling factor (default {config.LAMBDA}).')
    parser.add_argument('--search_lambda', action='store_true', default=False,
                        help='Search lambda once per fire function (feature-MSE proxy).')
    args = parser.parse_args()
    main(args)