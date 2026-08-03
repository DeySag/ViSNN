import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from calibration.profile import collect_profiles, compute_channel_thresholds
from data import make_dataset
from losses.depth_loss import calculate_rmse, compute_depth_loss
from models.backbone import get_mobilenetv2_backbone
from models.decoder import SimpleDepthDecoder
from models.snn import convert_to_snn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_root(overridden):
    if overridden:
        return overridden
    return config.TARTANAIR_ROOT if config.DATASET == 'tartanair' else config.DATA_ROOT


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

    # 1. Procure continuous backbone
    backbone = get_mobilenetv2_backbone(device)

    # 2. Spatial-masked channel-wise calibration
    profiles = collect_profiles(backbone, val_loader, device, config.CALIBRATION_BATCHES)
    channel_thresholds = compute_channel_thresholds(profiles, config.PERCENTILE, config.CROP_MARGIN)
    print(f'Computed channel-wise thresholds for {len(channel_thresholds)} layers.')

    # 3. SNN conversion surgery
    convert_to_snn(backbone, channel_thresholds, device)
    print('Surgery complete. Backbone is now a strict T=1 Scale-and-Fire SNN.')

    # 4. Assemble full model, freeze the spiking encoder
    model = SimpleDepthDecoder(backbone).to(device)
    model.eval()

    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    for param in model.decoder.parameters():
        assert param.requires_grad, 'Decoder gradient is frozen!'

    # 5. Optimizer hooks up exclusively to the continuous decoder
    optimizer = optim.AdamW(model.decoder.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # 6. Training loop
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for images, gt_depths in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS}'):
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

    # 7. Evaluation
    trained_rmse = run_validation(model, val_loader, device, config.MAX_BATCHES)
    print(f'Trained T=1 SNN RMSE: {trained_rmse:.4f}')

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'trained_t1_snn_depth.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the T=1 SNN FastDepth decoder.')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root (overrides config DATA_ROOT/TARTANAIR_ROOT).')
    args = parser.parse_args()
    main(args)