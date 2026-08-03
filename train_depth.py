import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
from tqdm import tqdm

import config
from calibration.lambda_search import search_lambda
from calibration.profile import collect_profiles, compute_channel_thresholds
from data import make_dataset
from losses.depth_loss import calculate_rmse, compute_depth_loss
from models.backbone import get_mobilenetv2_backbone
from models.decoder import SimpleDepthDecoder
from models.snn import convert_to_snn
from models.spiking_encoder import SpikingEncoder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_root(overridden):
    if overridden:
        return overridden
    return config.TARTANAIR_ROOT if config.DATASET == 'tartanair' else config.DATA_ROOT


def calibrate_thresholds(backbone, val_loader, device):
    """Spatial-masked, channel-wise threshold calibration (continuous pass)."""
    profiles = collect_profiles(backbone, val_loader, device, config.CALIBRATION_BATCHES)
    channel_thresholds = compute_channel_thresholds(profiles, config.TOP_P, config.CROP_MARGIN)
    print(f'Computed channel-wise thresholds for {len(channel_thresholds)} layers.')
    return channel_thresholds


def build_spiking_encoder(device, timesteps, fire_fn, lambda_, n_levels,
                          channel_thresholds):
    """Fresh backbone + SNN surgery -> SpikingEncoder wrapper."""
    backbone = get_mobilenetv2_backbone(device)
    convert_to_snn(backbone, channel_thresholds, device, lambda_=lambda_,
                   fire_fn=fire_fn, timesteps=timesteps, n_levels=n_levels)
    print(f'Surgery complete: SFN ({fire_fn}) at T={timesteps}, lambda={lambda_}.')
    return SpikingEncoder(backbone.features, timesteps=timesteps)


def search_lambda_for_encoder(snn_encoder, device, val_loader):
    """Grid-search the global scaling factor via feature-MSE proxy."""
    golden = torchvision_models.mobilenet_v2(weights='DEFAULT').features.to(device)
    golden.eval()
    best, results = search_lambda(snn_encoder, golden, val_loader, device,
                                  grid=config.LAMBDA_SEARCH_GRID,
                                  num_batches=config.CALIBRATION_BATCHES)
    print(f'Lambda search: {results} -> best lambda={best:.3f}')
    return best


def train_decoder(model, train_loader, device, epochs=config.NUM_EPOCHS):
    """Train only the continuous decoder on top of the frozen spiking encoder."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    for param in model.decoder.parameters():
        assert param.requires_grad, 'Decoder gradient is frozen!'

    optimizer = optim.AdamW(model.decoder.parameters(),
                            lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for images, gt_depths in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            images = images.to(device)
            gt_depths = gt_depths.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                spiking_features = model.encoder(images)

            predicted_depths = model.decoder(spiking_features)
            loss = compute_depth_loss(predicted_depths, gt_depths, config.GRAD_LOSS_ALPHA)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f'Epoch {epoch + 1}: avg train loss = {epoch_loss / n_batches:.4f}')
    model.eval()


def run_validation(model, loader, device, limit):
    total_rmse = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for images, gt_depths in tqdm(loader, total=limit, desc='Validating'):
            images = images.to(device)
            gt_depths = gt_depths.to(device)
            total_rmse += calculate_rmse(model(images), gt_depths)
            n += 1
            if n >= limit:
                break
    return total_rmse / n


def main(args):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
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

    # 1. Calibrate thresholds on a continuous backbone (once, dataset stats).
    probe_backbone = get_mobilenetv2_backbone(device)
    channel_thresholds = calibrate_thresholds(probe_backbone, val_loader, device)
    del probe_backbone

    # 2. Build the spiking encoder with the requested SFN configuration.
    lambda_ = args.lambda_
    encoder = build_spiking_encoder(device, args.timesteps, args.fire_fn,
                                    lambda_, args.n_levels, channel_thresholds)
    encoder.eval()

    # 3. Optional global-lambda search on the validation feature-MSE proxy.
    if args.search_lambda:
        lambda_ = search_lambda_for_encoder(encoder, device, val_loader)

    # 4. Assemble the full depth model and train the decoder.
    model = SimpleDepthDecoder(encoder).to(device)
    model.eval()
    train_decoder(model, train_loader, device, epochs=args.epochs)

    # 5. Evaluate.
    trained_rmse = run_validation(model, val_loader, device, config.MAX_BATCHES)
    print(f'Trained SFN RMSE (T={args.timesteps}, {args.fire_fn}, lambda={lambda_}): {trained_rmse:.4f}')

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'trained_snn_depth.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the SFN FastDepth decoder.')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root (overrides config DATA_ROOT/TARTANAIR_ROOT).')
    parser.add_argument('--timesteps', type=int, default=config.TIMESTEPS,
                        help=f'Inference timesteps T (default {config.TIMESTEPS}).')
    parser.add_argument('--fire_fn', type=str, default=config.FIRE_FN,
                        choices=['binary', 'mtn'],
                        help=f'SFN fire function (default {config.FIRE_FN}).')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=config.LAMBDA,
                        help=f'Global SFN scaling factor (default {config.LAMBDA}).')
    parser.add_argument('--n_levels', type=int, default=config.N_LEVELS,
                        help=f'MTN quantization levels (default {config.N_LEVELS}).')
    parser.add_argument('--search_lambda', action='store_true', default=config.SEARCH_LAMBDA,
                        help='Grid-search lambda on the val feature-MSE proxy.')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help=f'Decoder training epochs (default {config.NUM_EPOCHS}).')
    args = parser.parse_args()
    main(args)