import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from config import DATA_PATH, SCALER_DIR, BATCH_SIZE

class IKDataset(Dataset):
    """
    PyTorch Dataset wrapping input (x,y) and target ([sinθ,cosθ] for each joint) arrays.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 22
):
    # Load raw CSV
    df = pd.read_csv(DATA_PATH)

    # planar inputs
    X = df[['x', 'y']].values

    # sin/cos targets for 3 joints → shape (N,6)
    Y_raw = df[['θ1', 'θ2', 'θ3']].values
    Y_sin = np.sin(Y_raw)
    Y_cos = np.cos(Y_raw)
    Y = np.concatenate([Y_sin, Y_cos], axis=1)

    # Train / temp split
    temp_frac = test_size + val_size
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=temp_frac, random_state=random_state
    )

    # Validation / test split
    val_frac = val_size / temp_frac
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=val_frac, random_state=random_state
    )

    # Prepare & load/fix X-scaler only
    SCALER_DIR.mkdir(exist_ok=True)
    scaler_X_path = SCALER_DIR / 'scaler_X.pkl'
    if scaler_X_path.exists():
        scaler_X = joblib.load(scaler_X_path)
    else:
        scaler_X = StandardScaler().fit(X_train)
        joblib.dump(scaler_X, scaler_X_path)

    # Transform X splits
    X_train = scaler_X.transform(X_train)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_test)

    # Leave Y (sin/cos) as-is
    # Y_train, Y_val, Y_test are already in [-1,1]
    
    # Build PyTorch loaders
    train_ds = IKDataset(X_train, Y_train)
    val_ds   = IKDataset(X_val,   Y_val)
    test_ds  = IKDataset(X_test,  Y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
