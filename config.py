# ============================================
# Global configuration for the T=1 SNN pipeline
# ============================================

# Paths
DATA_ROOT = 'data/kitti'
CHECKPOINT_DIR = 'checkpoints'

# Spatial-Masked Calibration
PERCENTILE = 99.0
CROP_MARGIN = 2
CALIBRATION_BATCHES = 20

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