from pathlib import Path

# project root
BASE_DIR = Path(__file__).resolve().parent.parent

# docs folder
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)
DATA_PATH = DOCS_DIR / "manipulability_dataset.csv"

# saved scalers 
SCALER_DIR = DOCS_DIR / "scalers"
SCALER_DIR.mkdir(exist_ok=True)

# model checkpoints
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Training hyper-parameters
BATCH_SIZE     = 128
LR             = 1e-1
WEIGHT_DECAY   = 0
NUM_EPOCHS     = 80
EARLY_STOPPING = 10

# Model architecture  
HIDDEN_SIZES   = [512, 512, 512, 512, 256]
DROPOUT        = 0.0

# (Future, may not be necessary) multi-objective fine-tune weights
ALPHA          = 1.0
BETA           = 0.01
