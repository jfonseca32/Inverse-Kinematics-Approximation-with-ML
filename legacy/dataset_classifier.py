import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from config_classifier import DATA_PATH, BATCH_SIZE, SEED

TRAIN_FRAC = 5/7
VAL_FRAC   = 1/7
TEST_FRAC  = 1/7

class WorkspaceDataset(Dataset):
    """
    PyTorch Dataset for (x,y) to label data.
    Expects a CSV with columns ['x','y','label'].
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[['x','y']].values, dtype=torch.float32)
        self.y = torch.tensor(df['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(batch_size: int = BATCH_SIZE):
    """
    Load the full dataset, split into train/val/test by the fixed fractions,
    and return DataLoaders. Splitting is randomized with a fixed seed.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = WorkspaceDataset(DATA_PATH)
    N = len(dataset)

    # compute split sizes
    train_size = int(TRAIN_FRAC * N)
    val_size   = int(VAL_FRAC   * N)
    test_size  = N - train_size - val_size

    # random split with reproducibility
    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
