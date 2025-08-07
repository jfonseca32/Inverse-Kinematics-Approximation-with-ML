from pathlib import Path

# data paths
DATA_PATH    = Path(__file__).parent.parent/'docs'/'classification_dataset.csv'
OUTPUT_DIR   = Path(__file__).parent.parent/'checkpoints'

# model/training hyperparams
BATCH_SIZE   = 64
LR           = 4e-3
NUM_EPOCHS   = 100
HIDDEN_SIZE  = 128
SEED         = 42
