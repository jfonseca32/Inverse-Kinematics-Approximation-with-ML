"""
Count the number of each label in the 6-DOF classification dataset,
and for unreachable points (label 0), count how many are inside vs outside
the conservative workspace sphere.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Same DH_PARAMS
DH_PARAMS = [
    (    0.0,     0.0, 0.6718,  0.0),
    (-np.pi/2, 0.4318, 0.0,     0.0),
    (    0.0,  0.0203, 0.0,     0.0),
    (-np.pi/2,    0.0, 0.1500,  0.0),
    ( np.pi/2,    0.0, 0.0,     0.0),
    (-np.pi/2,    0.0, 0.0,     0.0),
]

# Conservative reach bound 
REACH = sum(np.hypot(a, d) for (_, a, d, _) in DH_PARAMS)

def main():
    # Locate and load the CSV
    script_dir = Path(__file__).parent
    docs_dir   = script_dir.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    csv_path   = docs_dir / "6DOF_classification_dataset.csv"
    df = pd.read_csv(csv_path)

    # Count of each label
    counts = df['label'].value_counts().sort_index()
    print("Overall label counts:")
    for lbl in [0, 1, 2, 3]:
        cnt = counts.get(lbl, 0)
        print(f"  Label {lbl}: {cnt}")

    # For unreachable (label=0), count inside vs outside sphere
    df_unreach = df[df['label'] == 0]
    positions = df_unreach[['x', 'y', 'z']].values
    dists = np.linalg.norm(positions, axis=1)

    inside  = np.count_nonzero(dists <= REACH)
    outside = np.count_nonzero(dists >  REACH)

    print(f"\nUnreachable breakdown (total {len(df_unreach)}):")
    print(f"  Inside sphere (≤ {REACH:.3f} m):  {inside}")
    print(f"  Outside sphere (> {REACH:.3f} m): {outside}")

if __name__ == "__main__":
    main()
