# ViSNN — Minimal-Timestep Spiking Neural Network Depth Estimation

Low-latency, power-efficient depth estimation for autonomous robotics by converting a continuous **MobileNetV2** backbone into a **Scale-and-Fire SNN (SFN)** at the **minimum timestep where the accuracy/energy tradeoff is justified**, trained to predict `224×224` depth maps via a continuous upsampling decoder.

**Research anchor:** "One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons" (arXiv:2510.23383). The SFN neuron, scaling factor λ, and MTN fire function are implemented per that paper.

## What we intend to build

1. **ANN → SNN conversion at minimal T, not strict T=1.** We replace the continuous MobileNetV2 encoder with an SFN (per-channel thresholds) and **sweep the timestep T** to find the knee of the accuracy-vs-energy curve. T=1 is the low-accuracy extreme; the goal is the smallest T whose depth RMSE is still acceptable.
2. **A proper SFN, not a λ=1 binary step.** Two mechanisms from the paper:
   - **Global scaling factor λ ∈ (0,1]** (searched on the validation feature-MSE proxy) — the decisive accuracy lever.
   - **Two fire functions, ablated:** `binary` (classic scale-and-fire) and `mtn` (multi-threshold neuron: `θ·clip(⌊h/θ⌋, 0, N)`).
3. **T-sweep Pareto harness.** For T ∈ {1,2,4,8,16,32} × fire_fn ∈ {binary, mtn}: calibrate → convert → train decoder → report RMSE, feature MSE, and a theoretical MAC→AC energy ratio. Emits `results/sweep_results.csv` + `results/sweep_plot.png`.
4. **Dataset-agnostic pipeline.** The identical machinery runs on KITTI or TartanAir via a config flip.

## Repository Layout

```
ViSNN/
├── config.py                 # Global hyperparameters + SFN/T-sweep settings
├── train_depth.py            # Single-config training (--timesteps/--fire_fn/--lambda)
├── sweep_timesteps.py        # T x fire_fn Pareto sweep -> results CSV + plot
├── energy.py                 # Theoretical MAC->AC energy estimator
├── requirements.txt
├── reference/                # Prior prototype notebook (reference only)
├── fastdepth-t1-snn-prototype.ipynb  # Cell-by-cell pipeline prototype
├── data/
│   ├── __init__.py           # Dataset registry + make_dataset factory + transforms
│   ├── kitti.py              # KITTIDepthDataset + align_depth_target
│   └── tartanair.py          # TartanAirDataset + align_depth_target
├── models/
│   ├── backbone.py           # make_snn_ready + get_mobilenetv2_backbone
│   ├── snn.py                # SFNNeuron + convert_to_snn + set_lambda/reset
│   ├── spiking_encoder.py    # T-step rate-averaging encoder wrapper
│   └── decoder.py            # SimpleDepthDecoder (continuous upsampling head)
├── losses/
│   └── depth_loss.py         # calculate_rmse, compute_depth_loss, compute_multibox_loss
└── calibration/
    ├── profile.py            # collect_profiles + compute_channel_thresholds
    └── lambda_search.py      # search_lambda (global λ via feature-MSE proxy)
```

## Pipeline (stages)

1. **Backbone procurement & surgery prep:** Load pretrained `mobilenet_v2`, recursively replace `ReLU6` → `nn.ReLU(inplace=False)` so hooks can read clean voltages.
2. **Spatial-masked channel-wise calibration:** Run forward hooks over multiple batches, crop the outer padding artifacts (`CROP_MARGIN=2`), and compute a **unique threshold per channel** at the `TOP_P` percentile of the uncorrupted spatial core. **Performed once** and reused across the sweep.
3. **SFN conversion surgery:** Recursively swap every `nn.ReLU` for an `SFNNeuron`:
   - `thresholds` registered buffer `[1, C, 1, 1]` (broadcastable);
   - **T=1:** `o = λθ · G_{λθ}(h)` — `binary` = step at effective threshold `λθ`; `mtn` = `λθ·clip(⌊h/(λθ)⌋, 0, N)`.
   - **T>1:** membrane accumulation + reset-by-subtraction; the `SpikingEncoder` duplicates input across T steps and returns the **rate-averaged** features.
4. **Global λ search (optional):** `search_lambda` grid-searches λ ∈ `LAMBDA_SEARCH_GRID` on the validation feature-MSE proxy (per fire function).
5. **Encoder freeze + decoder training:** `requires_grad = False` on the SNN encoder; AdamW (`lr=1e-4`, `wd=1e-4`) trains only the continuous decoder. Spike features are extracted under `torch.no_grad()`, then the decoder maps them to depth.
6. **Evaluation:** depth RMSE vs ground truth, feature-level MSE vs a fresh continuous encoder, and the `energy.py` MAC→AC ratio.

## The SFN neuron (arXiv:2510.23383)

```
o(t) = λθ · G_{λθ}(h(t))
θ   = top-p% activation per channel (calibrated)
λ   = global scaling factor ∈ (0, 1]  (searchable)
G   = fire function:
      binary : step at λθ            -> o = λθ·1[h ≥ λθ]
      mtn    : θ·clip(⌊h/θ⌋, 0, N)   -> multi-level quantization
```

Key facts from the paper driving our design:
- A single-timestep Multi-Threshold Neuron is **theoretically equivalent** to a multi-timestep IF neuron (Temporal-to-Spatial Equivalence Theory).
- λ is decisive: without scaling, all fire functions collapse (<5% accuracy); with it, near-lossless T=1.
- The paper reports 88.8% ImageNet-1K, and on **COCO-2017 detection 60.3 mAP@.5:.95 at T=1** — the justification for Track B later.

## Usage

```bash
# Single configuration
python train_depth.py --data_root <path> \
    --timesteps 4 --fire_fn mtn --lambda 0.25 --search_lambda

# Full accuracy/energy sweep
python sweep_timesteps.py --data_root <path> --epochs 5 [--search_lambda]

# Outputs -> results/sweep_results.csv, results/sweep_plot.png
```

The dataset is selected via `DATASET` in `config.py` (`'kitti'` or `'tartanair'`); the data root falls back to `DATA_ROOT` / `TARTANAIR_ROOT` unless overridden with `--data_root`.

## Loss

```python
compute_depth_loss(pred, target, alpha=0.1):
    rmse   = sqrt(MSE(pred, target))
    grad_x = mean(|pred[:, :, :, :-1] - pred[:, :, :, 1:]|)
    grad_y = mean(|pred[:, :, :-1, :] - pred[:, :, 1:, :]|)
    return rmse + alpha * (grad_x + grad_y)
```

> Note: the reference loss `sqrt(mean((pred - target) * 2))` was **incorrect** (missing the square). The implementation uses the corrected `sqrt(MSE(...))`.

## Energy estimation (`energy.py`)

SNN inference replaces multiply-accumulate (MAC) with accumulate-only (AC) ops + a threshold compare. Using 45nm figures (Horowitz 2014; `E_MAC=4.6pJ`, `E_AC=0.9pJ`, `E_COMPARE=0.05pJ`), `estimate_energy(model, input)` returns `(ann_energy_pJ, snn_energy_pJ, ac_fraction, firing_rate)`. **Energy is a hardware property** — these are relative estimates for the tradeoff curve, not deployment power.

## Dataset Merge (KITTI ↔ TartanAir)

Both datasets plug into the **identical** pipeline — models, SNN surgery, calibration, losses, and training loop are dataset-agnostic, consuming `[B,3,224,224]` images + `[B,1,224,224]` depth tensors.

### Supported layouts

**KITTI** (`data/kitti/`):
```
<data_root>/
├── image/   # RGB images (.png/.jpg/.jpeg)
└── depth/   # 16-bit PNG dense LiDAR; meters = pixel / 256.0
```
Transform: `CenterCrop(224×224)` on RGB and aligned depth.

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
3. Run `python train_depth.py` or `python sweep_timesteps.py`.

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
| `DATA_ROOT` / `TARTANAIR_ROOT` | `data/kitti` / `data/tartanair` | Dataset roots |
| `CHECKPOINT_DIR` / `RESULTS_DIR` | `checkpoints` / `results` | Output dirs |
| `TOP_P` | `99.0` | Per-channel threshold percentile (top-p%) |
| `CROP_MARGIN` | `2` | Padding-artifact margin cropped during calibration |
| `CALIBRATION_BATCHES` | `20` | Batches for threshold stabilisation |
| `TIMESTEPS` | `1` | Inference timesteps T |
| `FIRE_FN` | `binary` | SFN fire function (`binary` or `mtn`) |
| `LAMBDA` | `1.0` | Global SFN scaling factor |
| `N_LEVELS` | `8` | MTN quantization levels |
| `LAMBDA_SEARCH_GRID` | `(0.1,0.25,0.5,0.75,1.0)` | λ candidates for search |
| `SEARCH_LAMBDA` | `False` | Grid-search λ on val |
| `SWEEP_TIMESTEPS` | `[1,2,4,8,16,32]` | T-sweep range |
| `SWEEP_FIRE_FNS` | `['binary','mtn']` | Fire functions swept |
| `SWEEP_EPOCHS` | `5` | Decoder epochs per sweep config |
| `MAX_BATCHES` | `100` | Validation batch cap |
| `INPUT_SIZE` | `224` | Spatial input size |
| `BATCH_SIZE` | `16` | Batch size |
| `LR` / `WEIGHT_DECAY` | `1e-4` / `1e-4` | AdamW hyperparameters |
| `GRAD_LOSS_ALPHA` | `0.1` | Spatial-gradient loss weight |
| `NUM_EPOCHS` | `10` | Decoder training epochs (single-config) |
| `SEED` | `42` | Reproducibility seed |

## Key Modules

- **`SFNNeuron`** (`models/snn.py`): per-channel threshold buffer, `λ` scaling, `binary`/`mtn` fire functions, and a T>1 membrane-accumulation path.
- **`SpikingEncoder`** (`models/spiking_encoder.py`): runs a frozen feature extractor over T timesteps and returns rate-averaged features (T=1 = single pass).
- **`convert_to_snn`** (`models/snn.py`): recursive ReLU → `SFNNeuron` surgery.
- **`set_lambda` / `reset_spiking_state`** (`models/snn.py`): global λ updates and per-batch membrane resets.
- **`collect_profiles` / `compute_channel_thresholds`** (`calibration/profile.py`): spatial-masked channel-wise threshold calibration.
- **`search_lambda`** (`calibration/lambda_search.py`): global λ grid search on a feature-MSE proxy.
- **`estimate_energy`** (`energy.py`): MAC→AC energy-ratio estimator.
- **`SimpleDepthDecoder`** (`models/decoder.py`): continuous upsampling block (1280→256→128→64→1) to `224×224`.

## Roadmap (parallel tracks)

- **Track A — Depth (FastDepth):** This repo. KITTI (or TartanAir) dense-depth, RMSE-optimized, with the SFN T-sweep as the core result.
- **Track B — Detection (SSD/COCO):** Binarized MobileNet feature extractor + continuous regression/classification heads. The SFN paper reports 60.3 mAP@.5:.95 on COCO-2017 at T=1, making this tractable; `compute_multibox_loss` scaffolding already exists in `losses/depth_loss.py`, with prior-box matching + SSD heads planned.
