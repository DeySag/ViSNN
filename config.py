# ============================================
# Global configuration for the SNN depth pipeline
# ============================================

# Dataset selector: 'kitti' | 'tartanair'
DATASET = 'kitti'

# Paths
DATA_ROOT = 'data/kitti'
TARTANAIR_ROOT = 'data/tartanair'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'

# Spatial-Masked Calibration
TOP_P = 99.0                # threshold percentile per channel (top-p%)
CROP_MARGIN = 2             # padding-artifact margin cropped during calibration
CALIBRATION_BATCHES = 20

# SFN (Scale-and-Fire Neuron) settings
TIMESTEPS = 1               # inference timesteps (T)
FIRE_FN = 'binary'          # 'binary' | 'mtn' (multi-threshold neuron)
LAMBDA = 1.0                # global scaling factor (paper: ~0.25)
N_LEVELS = 8                # MTN quantization levels
LAMBDA_SEARCH_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)
SEARCH_LAMBDA = False       # grid-search lambda on the val set

# T-sweep harness
SWEEP_TIMESTEPS = [1, 2, 4, 8, 16, 32]
SWEEP_FIRE_FNS = ['binary', 'mtn']
SWEEP_EPOCHS = 5

# Evaluation
MAX_BATCHES = 100

# Training
INPUT_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_LOSS_ALPHA = 0.1

# Reproducibility
SEED = 42