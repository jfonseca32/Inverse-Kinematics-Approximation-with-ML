import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import DATA_PATH
from utils import compute_manipulability_torch


def load_data():
    """
    Load the raw dataset CSV into a DataFrame and numpy arrays.

    Returns:
        df (pd.DataFrame): full dataset
        X (np.ndarray): shape (N,3) array of [x,y,z]
        Y (np.ndarray): shape (N,4) array of [θ1,θ2,θ3,θ4]
    """
    df = pd.read_csv(DATA_PATH)
    X = df[['x', 'y', 'z']].values
    Y = df[['θ1', 'θ2', 'θ3', 'θ4']].values
    return df, X, Y


def check_one_to_one(df, tol=1e-6):
    """
    Check for one-to-one mapping of input (x,y,z) to exactly one output θ.
    Rounds inputs to a tolerance and reports any duplicates.

    Args:
        df (pd.DataFrame): dataset containing 'x','y','z','θ1'...'θ4'
        tol (float): rounding tolerance for input grouping

    Returns:
        duplicates (pd.DataFrame): subset of df where duplicate inputs occur
    """
    # Determine decimal places for rounding
    decimals = int(-np.log10(tol)) if tol < 1 else 0
    df_r = df.copy()
    df_r['x_r'] = df_r['x'].round(decimals)
    df_r['y_r'] = df_r['y'].round(decimals)
    df_r['z_r'] = df_r['z'].round(decimals)

    # Find groups with more than one entry
    grouped = df_r.groupby(['x_r', 'y_r', 'z_r'])
    duplicates = grouped.filter(lambda g: len(g) > 1)

    if duplicates.empty:
        print("✅ No duplicate (x,y,z) entries found; mapping is one-to-one.")
    else:
        groups = duplicates.groupby(['x_r', 'y_r', 'z_r'])
        print(f"⚠️ Found {duplicates.shape[0]} rows for {groups.ngroups} duplicate input locations:")
        for (xr, yr, zr), grp in groups:
            print(f" Input ~({xr}, {yr}, {zr}) has {len(grp)} labels:")
            print(grp[['x','y','z','θ1','θ2','θ3','θ4']])
    return duplicates


def compute_manipulability(Y):
    """
    Compute manipulability metric w(θ) = sqrt(det(J J^T)) for each θ in Y.

    Args:
        Y (np.ndarray): shape (N,4) joint angles in radians

    Returns:
        w (np.ndarray): shape (N,) manipulability values
    """
    with torch.no_grad():
        thetas = torch.from_numpy(Y.astype(np.float32))
        w = compute_manipulability_torch(thetas)
    return w.cpu().numpy()


def check_label_manipulability(Y, thresh=1e-2):
    """
    Compute manipulability for each label and report basic statistics.
    Flags any samples below a threshold.

    Args:
        Y (np.ndarray): shape (N,4) dataset labels
        thresh (float): threshold below which manipulability is considered low

    Returns:
        w (np.ndarray): computed manipulability values
        low_idx (np.ndarray): indices where w < thresh
    """
    w = compute_manipulability(Y)
    print("Manipulability stats:")
    print(f"  min = {w.min():.6f}")
    print(f"  max = {w.max():.6f}")
    print(f"  mean = {w.mean():.6f}")
    print(f"  std = {w.std():.6f}")
    low_idx = np.where(w < thresh)[0]
    if low_idx.size > 0:
        print(f"⚠️ {len(low_idx)} samples have manipulability below {thresh}.")
    else:
        print(f"✅ All {len(w)} samples have manipulability ≥ {thresh}.")
    return w, low_idx


def plot_xyz_scatter(X):
    """
    Plot the input positions (x,y,z) in 3D space.

    Args:
        X (np.ndarray): shape (N,3) array of inputs
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Scatter of Input Coordinates')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df, X, Y = load_data()
    # One-to-one mapping check
    dup = check_one_to_one(df)
    # Label correctness via manipulability stats
    w_vals, low = check_label_manipulability(Y)
    # 3D scatter plot
    # plot_xyz_scatter(X)
