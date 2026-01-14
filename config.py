from pathlib import Path

# Directory paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'Data'
MODELS_DIR = ROOT_DIR / 'Models'
RESULTS_DIR = ROOT_DIR / 'Results'
SRC_DIR = ROOT_DIR / 'src'

# Model hyperparameters
LATENT_DIM = 20
TEXT_DIM = 512
ACTION_TIME_STEPS = 32
DIM_ACTION = 24
EPOCHS = 150
BATCH_SIZE = 32

# Loss weights
ALPHA = 1   # Adversarial loss weight
BETA = 10   # Distance loss weight
GAMMA = 5   # Velocity loss weight

# Optimizer settings
LEARNING_RATE_G = 0.00002
LEARNING_RATE_D = 0.00002

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15

# Label smoothing ranges
REAL_LABEL_MIN = 0.9
REAL_LABEL_MAX = 1.0
FAKE_LABEL_MIN = 0.0
FAKE_LABEL_MAX = 0.1

# Checkpoint settings
CHECKPOINT_FREQUENCY = 50
