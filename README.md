# ViSNN — Two-Track ANN→SNN Conversion for Depth & Detection

Converts pretrained CNN backbones into **single-timestep (T=1) spiking networks**
and trains continuous task heads on top of the frozen spiking features.

| | Track A | Track B |
|---|---|---|
| **Task** | Monocular depth estimation | Object detection |
| **Dataset** | KITTI (dense LiDAR depth annotations) | COCO 2017 |
| **Architecture** | FastDepth-style: MobileNetV2 / ResNet-50 encoder + upsampling decoder | SSD (MultiBox) on a MobileNetV2 trunk |
| **Spiking part** | Whole encoder (frozen) | MobileNet stages (frozen) |
| **Trained part** | Depth decoder | SSD extra layers + loc/cls heads |
| **Objective** | Masked RMSE + spatial gradient | Smooth-L1 + cross-entropy (MultiBox) |
| **Metric** | RMSE, AbsRel, δ1/δ2/δ3 | mAP@0.5, mAP@[.5:.95] |

Based on *"One-Timestep is Enough: Achieving High-performance ANN-to-SNN
Conversion via Scale-and-Fire Neurons"* (arXiv:2510.23383), and on the
[DeySag/ViSNN](https://github.com/DeySag/ViSNN) reference implementation.

### References

Local copies of the key references live in the `reference/` folder:

- **`reference/2510.23383v1 (1).pdf`** — the *"One-Timestep is Enough"* paper. This
  is the primary research anchor for the project: the **Scale-and-Fire neuron**
  itself, the global **scaling factor λ**, and the **multi-threshold (MTN)** fire
  function are all implemented per this paper (see `models/snn.py` and
  `calibration/lambda_search.py`).
- **`reference/fastdepth-t1-conversion (1).ipynb`** — the original prototype
  notebook that motivated the conversion approach, kept for provenance.

---

## Quick start

Install (CPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then the rest:

```bash
pip install -r requirements.txt
```

Verify the whole project without downloading a single dataset — 40 checks
covering data alignment, neuron semantics, losses, box maths, metrics, and both
end-to-end tracks:

```bash
python tests/test_pipeline.py
```

Run either track on generated stand-in data:

```bash
python train_depth.py --synthetic --epochs 6 --fire-fn mtn --lambda 0.25 --lr 1e-3
```

```bash
python train_ssd.py --synthetic --epochs 12 --fire-fn mtn --lambda 0.25 --lr 1e-3
```

> **Note on the defaults.** The default configuration is strict binary spikes
> at λ=1.0 — a genuine 1-bit activation that does not carry enough information
> to generalise (measured val mAP 0.017 vs 0.819 for the working
> configuration). The defaults in `config.py` are retained as the experimental
> baseline; the two flags above select the configuration that works. Full
> measurements are in [Measured results](#measured-results).

---

## Running on the real datasets

### Track A — KITTI

Download the **depth prediction** annotated maps *and* the matching raw frames
(the annotation archive contains no RGB images). Point `--data-root` at any
directory that contains both — the loader walks the tree, so the nesting does
not matter:

```
datasets/kitti/
├── 2011_09_26_drive_0001_sync/
│   ├── image_02/data/0000000005.png                    # RGB
│   └── proj_depth/groundtruth/image_02/0000000005.png  # 16-bit depth
└── ...
```

```bash
python train_depth.py --data-root datasets/kitti --epochs 10 --batch-size 16
```

### Track B — COCO

Standard COCO 2017 layout:

```
datasets/coco/
├── annotations/instances_train2017.json
├── annotations/instances_val2017.json
├── train2017/*.jpg
└── val2017/*.jpg
```

```bash
python train_ssd.py --data-root datasets/coco --epochs 20 --lr 1e-3
```

`pycocotools` is **not** required — the annotation JSON is parsed with the
standard library and mAP is implemented directly.

---

## Project layout

```
config.py              All hyperparameters and paths, one place
utils.py               Seeding, device, checkpoints, meters
pipeline.py            Calibrate -> convert -> freeze, for both tracks
train_depth.py         Track A entrypoint
train_ssd.py           Track B entrypoint

data/
├── kitti.py           os.walk indexing, 16-bit depth decode, drive-disjoint split
├── coco.py            stdlib JSON parser, sparse->dense category remap
├── transforms.py      Geometrically-aligned RGB/depth transforms
├── synthetic.py       KITTI/COCO-shaped generated data (no download)
└── loaders.py         DataLoader construction, real<->synthetic switch

models/
├── snn.py             StrictT1SFN neuron + recursive conversion
├── backbone.py        MobileNetV2 / ResNet-50, multi-scale SSD trunk
├── decoder.py         SimpleDepthDecoder (32x upsampling)
├── ssd.py             PriorBox, SSD heads, decode + NMS
└── box_utils.py       IoU, encode/decode, prior matching, NMS

calibration/
├── profile.py         Spatial-masked per-channel threshold calibration
└── lambda_search.py   Global scaling-factor grid search

losses/
├── depth_loss.py      Masked RMSE + spatial gradient
└── multibox_loss.py   MultiBox with hard-negative mining

validation/
├── metrics_depth.py   RMSE, AbsRel, SqRel, RMSE_log, delta1/2/3
├── metrics_detection.py  mAP from first principles
└── visualize.py       Depth triptychs, box overlays, curves, spike rates

tests/test_pipeline.py 40 end-to-end correctness checks
```

---

## Pipeline

**1. Data.** `DepthJointTransform` applies one crop to both the RGB frame and
its depth map, so geometry cannot drift. COCO boxes are normalized to
fractional `xyxy` at parse time, which makes them invariant to the later resize.

**2. Calibration and conversion.** The continuous pretrained backbone is run
over ~20 batches while forward hooks record per-channel activation percentiles,
excluding a 2-pixel border (zero-padded convolutions inflate the outermost
pixels). Every activation is then replaced by a `StrictT1SFN`:

```
o = θ_eff · 1[x ≥ θ_eff],    θ_eff = λ · θ_c
```

Emitting `θ_eff` rather than a bare `1.0` preserves the activation magnitude the
next convolution expects, which is what makes T=1 conversion work. The converted
trunk is then frozen.

**3. Training.** AdamW sees only the continuous head. The frozen trunk runs
under `torch.no_grad()`, so the computational graph starts at the decoder / SSD
extras.

**4. Validation.** Metrics accumulate over the whole validation set before the
final division, figures are written per epoch, and per-layer spike rates are
reported as the energy proxy.

---

## Key options

| Flag | Meaning |
|---|---|
| `--synthetic` | Generated data; no dataset download needed |
| `--timesteps N` | T. `1` is the headline config; `N>1` accumulates a membrane and averages rate-coded outputs |
| `--fire-fn {binary,mtn}` | Strict binary spike, or multi-threshold graded spikes |
| `--lambda L` | Global threshold scale. Lower ⇒ more spikes, more information, more energy |
| `--search-lambda` | Grid-search λ against val RMSE / mAP |
| `--percentile P` | Calibration percentile (default 99.0) |
| `--crop-margin N` | Border pixels excluded from calibration (default 2) |
| `--backbone {mobilenet_v2,resnet50}` | Track A encoder |
| `--gradient-mode {smoothness,matching}` | See "Notes on the spec" below |
| `--limit-train / --limit-val` | Subsample for fast iteration |

---

## Measured results

Everything below was executed on this machine, not assumed.

### Correctness

- **40/40 checks pass** in `tests/test_pipeline.py`, including miniature
  on-disk KITTI and COCO fixtures in the real formats (16-bit depth PNGs,
  sparse COCO category ids, `iscrowd` filtering).
- **Oracle check** — feeding exact encoded targets through `decode → NMS → mAP`
  yields **mAP = 1.0**, so the evaluation path contributes no error of its own.
- **Fit check** — the SSD heads reach **mAP@0.5 = 1.000** on a fitted batch
  through the frozen T=1 spiking trunk, confirming prior matching, MultiBox
  loss, box decoding, NMS and mAP are all wired correctly.
- **Freeze check** — encoder weights are byte-identical before and after
  training.

### Choosing the neuron: a controlled experiment

Training on 512 synthetic images for 12 epochs with a *frozen* trunk, held-out
val split, everything else identical:

| Backbone | spike rate | train mAP@0.5 | **val mAP@0.5** |
|---|---|---|---|
| Continuous (no conversion) | — | 0.52–0.66 | **0.66–0.73** |
| `binary`, λ = 1.00 *(spec default)* | 0.171 | 0.38 | **0.017** |
| `mtn` L=8, λ = 1.00 | 0.199 | 0.63 | **0.118** |
| `mtn` L=8, λ = 0.25 | 0.474 | 0.97 | **0.819** |

Same trend on the depth track (256 images, 6 epochs):

| Neuron | spike rate | val RMSE | AbsRel | δ1 |
|---|---|---|---|---|
| `binary`, λ = 1.00 | 0.232 | 5.93 m | 0.185 | 0.745 |
| `mtn` L=8, λ = 1.00 | 0.258 | 5.54 m | 0.173 | 0.787 |
| `mtn` L=8, λ = 0.25 | 0.514 | **4.08 m** | **0.111** | **0.864** |

**Reading this.** With strict binary spikes the model *memorises* the training
set (train mAP 0.38) but does not generalise (val mAP 0.017). The continuous
control is the important row: same architecture, same frozen trunk, same data,
same schedule — it generalises fine. That rules out any data, matching, or
metric bug and isolates the cause to the quantization itself.

A binary spike is a genuine 1-bit activation: every input above threshold
collapses to the same value, so all magnitude information is destroyed. The
multi-threshold neuron emits `floor(x/θ)` clipped to L levels — about 3 bits —
and λ = 0.25 lowers the thresholds so those levels are actually exercised
rather than everything saturating at level 0 or 1. Note that λ and the neuron
must be tuned *together*: lowering λ under a binary neuron makes things worse
(mAP 0.187 at λ=0.25), because it fires more neurons without adding any
resolution.

λ ≈ 0.25 is also the optimum reported in the Scale-and-Fire paper, so this
reproduces the published result. `--search-lambda` runs the grid automatically.

---

## Design notes

A few places where the most direct formulation is wrong, and how the code
handles them.

1. **RMSE squaring.** `(predicted - target) * 2` doubles the error instead of
   squaring it, making the "RMSE" the square root of a signed mean, which
   returns `NaN` whenever the model over-predicts on average. Implemented as
   `** 2` in `losses/depth_loss.py`, with a regression test.

2. **Sparse LiDAR targets.** KITTI ground truth stores "no return" as `0.0`.
   Averaging over those zeros trains the network to predict 0 metres across
   ~85% of the frame, so every loss term is masked by `target > DEPTH_MIN`.

3. **MultiBox balancing.** SSD emits ~1194 predictions per image for a handful
   of objects. Averaging localisation over all priors drowns the real targets,
   and averaging classification over ~99% background converges to "predict
   background everywhere" with mAP pinned at zero. `losses/multibox_loss.py`
   restricts localisation to matched positives and applies 3:1 hard-negative
   mining.

4. **Activation conversion.** torchvision's MobileNetV2 uses `nn.ReLU6`, which
   subclasses `nn.Hardtanh` rather than `nn.ReLU`; an `isinstance(nn.ReLU)`
   check would convert zero layers on that backbone. `models/snn.py`
   handles both.

Two further issues that were silent rather than wrong:

- **BatchNorm drift.** `requires_grad = False` does not stop BatchNorm updating
  its running statistics, so a plain `model.train()` keeps shifting the
  distribution the thresholds were calibrated against. Both models override
  `train()` to pin the frozen trunk to `eval()`.
- **ResNet activation sharing.** `Bottleneck.forward` calls one `self.relu`
  module three times, on tensors with two different channel counts, so
  per-channel thresholds cannot be assigned. `split_resnet_relus` gives each
  activation site its own module first.

One deliberate divergence, left configurable:

- The gradient term takes gradients of the *prediction alone*, which is a
  smoothness prior — it penalises all structure and blurs edges, the opposite
  of sharpening boundaries. It remains the default (`--gradient-mode
  smoothness`); `--gradient-mode matching` compares prediction gradients
  against ground-truth gradients and is the formulation that actually sharpens.

Also worth noting: a 224x224 `CenterCrop` of a 1242x375 KITTI frame discards
~82% of the width. It preserves true metric pixel scale, but `--spatial-mode
resize` keeps the full field of view if you want to ablate it.

---

## Outputs

```
checkpoints/<tag>_best.pt          best-metric weights (head only — the trunk is frozen)
checkpoints/<tag>_thresholds.npz   calibrated per-channel thresholds
results/<tag>_epoch<NN>.png        per-epoch qualitative figures
results/<tag>_curves.png           loss / metric curves
results/<tag>_spike_rates.png      per-layer firing rates
results/<tag>_summary.json         full run summary
```
