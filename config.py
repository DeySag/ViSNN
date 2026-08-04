"""Global configuration for the two-track SNN pipeline.

Track A -- FastDepth-style depth estimation on KITTI.
Track B -- SSD object detection on COCO.

Everything downstream reads from here, so a single edit re-targets both tracks.
Values can also be overridden per-run from the CLI (see train_depth.py /
train_ssd.py).
"""

import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Track A: expects the KITTI depth-prediction layout, i.e. somewhere under this
# root there are `image_02/data/*.png` RGB frames and matching
# `proj_depth/groundtruth/image_02/*.png` 16-bit depth maps. The loader walks
# the tree, so the exact nesting does not matter.
KITTI_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'kitti')

# Track B: expects the standard COCO layout:
#   COCO_ROOT/annotations/instances_train2017.json
#   COCO_ROOT/train2017/*.jpg
COCO_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'coco')
COCO_TRAIN_SPLIT = 'train2017'
COCO_VAL_SPLIT = 'val2017'

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ----------------------------------------------------------------------------
# Input geometry
# ----------------------------------------------------------------------------
INPUT_SIZE = 224            # both tracks run on 224x224 square tensors

# KITTI frames are ~1242x375. A raw 224x224 CenterCrop keeps true pixel scale
# but discards ~82% of the width. 'resize' keeps the full frame instead.
# 'crop' is the default; 'resize' is available for ablations.
KITTI_SPATIAL_MODE = 'crop'  # 'crop' | 'resize'

# ImageNet statistics -- the pretrained MobileNetV2 backbone expects these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# KITTI depth specifics
# ----------------------------------------------------------------------------
KITTI_DEPTH_SCALE = 256.0   # 16-bit PNG -> metres
DEPTH_MIN = 1e-3            # anything at/below this is "no LiDAR return"
DEPTH_MAX = 80.0            # KITTI evaluation cap, metres

# ----------------------------------------------------------------------------
# Spatial-masked calibration (per-channel SNN thresholds)
# ----------------------------------------------------------------------------
TOP_P = 99.0                # per-channel activation percentile -> threshold
CROP_MARGIN = 2             # border pixels dropped (conv padding artifacts)
CALIBRATION_BATCHES = 20    # batches streamed through the ANN while profiling

# ----------------------------------------------------------------------------
# Scale-and-Fire neuron (StrictT1SFN)
# ----------------------------------------------------------------------------
# Two fire functions are supported: strict binary spikes ('binary') and graded
# multi-threshold spikes ('mtn', floor(x/theta) clipped to N levels). The
# default is the binary neuron at an unscaled threshold (lambda = 1.0); this is
# the naive T=1 configuration and serves as the experimental baseline.
#
# A binary spike is a single bit, so all inputs above threshold map to the same
# value and magnitude information is lost. The graded neuron retains ~3 bits,
# and lowering lambda applies that resolution by lowering the effective
# thresholds. Measured on the detection track (identical architecture, data and
# schedule; 512 synthetic images, 12 epochs, held-out val split):
#
#   backbone                        spike rate   val mAP@0.5
#   continuous (no conversion)          --          0.66-0.73
#   binary,  lambda = 1.00             0.171        0.017      <- defaults
#   mtn L=8, lambda = 1.00             0.199        0.118
#   mtn L=8, lambda = 0.25             0.474        0.819      <- recommended
#
# lambda ~= 0.25 matches the optimum reported in the Scale-and-Fire paper. To
# use the recommended configuration, pass `--fire-fn mtn --lambda 0.25`.
TIMESTEPS = 1               # T=1 is the headline configuration
FIRE_FN = 'binary'          # 'binary' (strict T=1 spike) | 'mtn' (multi-level)
LAMBDA = 1.0                # global threshold scaling factor
N_LEVELS = 8                # quantization levels when FIRE_FN == 'mtn'
LAMBDA_SEARCH_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)
SEARCH_LAMBDA = False

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 0             # >0 on Linux; Windows spawn-start is slower here
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_LOSS_ALPHA = 0.1       # spatial-gradient term weight in the depth loss
GRAD_CLIP = 5.0

# ----------------------------------------------------------------------------
# SSD (Track B)
# ----------------------------------------------------------------------------
NUM_CLASSES = 81            # 80 COCO categories + background at index 0
SSD_NEG_POS_RATIO = 3       # hard-negative mining ratio
SSD_IOU_THRESHOLD = 0.5     # prior <-> ground-truth matching threshold
SSD_LOC_VARIANCES = (0.1, 0.2)
SSD_SCORE_THRESHOLD = 0.01  # detection score floor at inference
SSD_NMS_THRESHOLD = 0.45
SSD_TOP_K = 200

# ----------------------------------------------------------------------------
# Evaluation / logging
# ----------------------------------------------------------------------------
MAX_EVAL_BATCHES = 100      # cap validation length; None = full split
LOG_INTERVAL = 20           # batches between training log lines
VIS_SAMPLES = 4             # side-by-side figures written per validation pass

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42


def ensure_dirs():
    """Create the output directories the pipeline writes into."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
