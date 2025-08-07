from pathlib import Path

# data paths
DATA_PATH    = Path(__file__).parent.parent/'docs'/'6DOF_classification_dataset.csv'
OUTPUT_DIR   = Path(__file__).parent.parent/'checkpoints'

# model/training hyperparams
BATCH_SIZE   = 64
LR           = 8e-3
NUM_EPOCHS   = 200
HIDDEN_SIZE  = 256
SEED         = 42
