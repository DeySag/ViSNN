# ViSNN — Single-Timestep Spiking Neural Network Depth Estimation

Low-latency, power-efficient depth estimation for autonomous robotics by converting a continuous **MobileNetV2** backbone into a **strict single-timestep (T=1) Spiking Neural Network (SNN)**, trained to predict `224×224` depth maps via a continuous upsampling decoder.

## Overview

- **Objective:** Replace the continuous MobileNetV2 encoder with a T=1 Scale-and-Fire SNN while learning a continuous decoder that maps binary feature spikes to functional depth estimates.
- **Constraint:** Strict T=1 — no multi-timestep Leaky-Integrate-and-Fire (LIF) temporal loops.
- **Key result:** Spatial-masked, channel-wise threshold calibration dropped the Feature-Level MSE from **1.62 → 0.2303**, proving the spatial core geometry survives quantization without temporal recursion.

## Repository Layout

```
ViSNN/
├── config.py               # Global hyperparameters + paths
├── train_depth.py          # End-to-end pipeline entrypoint (argparse --data_root)
├── requirements.txt        # Python dependencies
├── reference/              # Prior prototype notebook (reference only)
└── fastdepth-t1-conversion (1).ipynb
├── fastdepth-t1-snn-prototype.ipynb  # Cell-by-cell pipeline prototype
├── data/
│   ├── __init__.py        # Dataset registry + make_dataset factory + transforms
│   ├── kitti.py           # KITTIDepthDataset + align_depth_target
│   └── tartanair.py       # TartanAirDataset + align_depth_target
├── models/
│   ├── backbone.py         # make_snn_ready + get_mobilenetv2_backbone
│   ├── snn.py              # StrictT1SFN + convert_to_snn surgery
│   └── decoder.py          # SimpleDepthDecoder
├── losses/
│   └── depth_loss.py       # calculate_rmse, compute_depth_loss, compute_multibox_loss
└── calibration/
    └── profile.py          # collect_profiles + compute_channel_thresholds
```

## Pipeline (6 stages)

1. **Backbone procurement & surgery prep:** Load pretrained `mobilenet_v2`, recursively replace `ReLU6` → `nn.ReLU(inplace=False)` so hooks can read clean voltages.
2. **Spatial-masked channel-wise calibration:** Run forward hooks over multiple batches, crop the outer padding artifacts (`CROP_MARGIN=2`), and compute a **unique threshold per channel** using the 99.0th percentile of the uncorrupted spatial core.
3. **SNN conversion surgery:** Recursively swap every `nn.ReLU` for a `StrictT1SFN`. Its `thresholds` buffer is registered as `[1, C, 1, 1]` for broadcasting; forward emits binary spikes `(x >= θ)` scaled by `θ` (`spikes * θ`).
4. **Encoder freeze:** `requires_grad = False` on the entire SNN backbone (protects calibrated thresholds); asserts decoder grads remain active.
5. **Decoder training:** Toroidal AdamW (`lr=1e-4`, `weight_decay=1e-4`) exclusively on the continuous decoder. Spikes are extracted under `torch.no_grad()`, then the decoder maps them to depth.
6. **Evaluation:** RMSE against ground truth + feature-level MSE vs a fresh continuous encoder.

## Loss

```python
compute_depth_loss(pred, target, alpha=0.1):
    rmse   = sqrt(MSE(pred, target))
    grad_x = mean(|pred[:, :, :, :-1] - pred[:, :, :, 1:]|)
    grad_y = mean(|pred[:, :, :-1, :] - pred[:, :, 1:, :]|)
    return rmse + alpha * (grad_x + grad_y)
```

> Note: the reference loss `sqrt(mean((pred - target) * 2))` was **incorrect** (missing the square). The implementation uses the corrected `sqrt(MSE(...))`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train_depth.py --data_root <path_to_data>
```

The dataset is selected via `DATASET` in `config.py` (`'kitti'` or `'tartanair'`); the data root falls back to `DATA_ROOT` / `TARTANAIR_ROOT` per dataset unless overridden with `--data_root`.

## Dataset Merge (KITTI ↔ TartanAir)

Both datasets plug into the **identical** T=1 pipeline — models, SNN surgery, calibration, losses, and training loop are dataset-agnostic, consuming `[B,3,224,224]` images + `[B,1,224,224]` depth tensors. Switching datasets only changes the loader + spatial transform.

### Supported layouts

**KITTI** (`data/kitti/`):
```
<data_root>/
├── image/   # RGB images (.png/.jpg/.jpeg)
└── depth/   # 16-bit PNG dense LiDAR; meters = pixel / 256.0
```
Transform: `CenterCrop(224×224)` on both RGB and aligned depth.

**TartanAir** (`data/tartanair/`):
```
<data_root>/
├── image_left/
│   └── **/image_left/*.png     # RGB
└── depth_left/
    └── **/depth_left/*.npy     # exact float meters (PNG /256.0 fallback)
```
Transform: `Resize(224×224)` on RGB and `F.interpolate` bilinear on depth.

### Switching between them

1. Set `DATASET = 'tartanair'` (or `'kitti'`) in `config.py`.
2. Optionally set `TARTANAIR_ROOT` / `DATA_ROOT`, or pass `--data_root`.
3. Run `python train_depth.py`.

### Adding a third dataset

1. Create `data/<name>.py` with a `Dataset` returning `image [3,224,224]`, `depth [1,224,224]`, plus an `align_depth_target` helper.
2. Register it in the `DATASET_CLASS` dict in `data/__init__.py`.
3. Add its spatial transform branch in `get_dataset_transform`.
4. Set `DATASET = '<name>'` in `config.py`.

The factory (`make_dataset`), calibration, surgery, and training loop require no further changes.

## Configuration (`config.py`)

| Key | Default | Purpose |
|-----|---------|---------|
| `DATASET` | `kitti` | Dataset selector (`'kitti'` or `'tartanair'`) |
| `DATA_ROOT` | `data/kitti` | KITTI root path |
| `TARTANAIR_ROOT` | `data/tartanair` | TartanAir root path |
| `PERCENTILE` | `99.0` | Channel-wise threshold percentile |
| `CROP_MARGIN` | `2` | Padding-artifact margin cropped during calibration |
| `CALIBRATION_BATCHES` | `20` | Batches for threshold stabilisation |
| `MAX_BATCHES` | `100` | Validation batch cap |
| `INPUT_SIZE` | `224` | Spatial input size |
| `BATCH_SIZE` | `16` | Batch size |
| `LR` | `1e-4` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `GRAD_LOSS_ALPHA` | `0.1` | Spatial-gradient loss weight |
| `NUM_EPOCHS` | `10` | Decoder training epochs |
| `SEED` | `42` | Reproducibility seed |

## Key Modules

- **`StrictT1SFN`** (`models/snn.py`): the T=1 Scale-and-Fire neuron with per-channel thresholds as a registered buffer.
- **`collect_profiles`** (`calibration/profile.py`): captures per-layer activations across a multi-batch pass.
- **`compute_channel_thresholds`** (`calibration/profile.py`): masks the padding artifacts and computes per-channel thresholds.
- **`convert_to_snn`** (`models/snn.py`): recursive ReLU → `StrictT1SFN` surgery.
- **`SimpleDepthDecoder`** (`models/decoder.py`): continuous upsampling block (1280→256→128→64→1) with bilinear scaling to `224×224`.

## Roadmap (parallel tracks)

- **Track A — Depth (FastDepth):** This repo. KITTI dense-depth, RMSE-optimized.
- **Track B — Detection (SSD):** COCO. Binarized MobileNet feature extractor + continuous regression/classification heads (`compute_multibox_loss` scaffolding provided in `losses/depth_loss.py`).