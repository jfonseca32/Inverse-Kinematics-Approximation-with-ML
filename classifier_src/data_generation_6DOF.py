"""
Generate a 6-DOF reachability/conditioning dataset for a Puma-style arm,
using a KD-tree prefilter to minimize expensive IK calls when sampling unreachable points.

Each row: [theta1…theta6, x, y, z, qx, qy, qz, qw, label]
  label 0 = unreachable
        1 = near-singular (bottom 10% κ_inv)
        2 = well-conditioned (middle 80%)
        3 = high-conditioned (top 10%)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.linalg import svd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# Dataset generation parameters
N_REACHABLE       = 400_000
N_UNREACHABLE     = 160_000
PERCENTILE_LOW    = 20
PERCENTILE_HIGH   = 80
TOL               = 1e-6
# fraction of REACH used as kd-tree distance threshold
NEIGHBOR_EPS_RATIO = 0.05  

# PUMA 560 DH parameters (alpha_{i-1}, a_{i-1}, d_i, theta_i_offset) ---
DH_PARAMS = [
    (    0.0,     0.0, 0.6718,  0.0),
    (-np.pi/2, 0.4318, 0.0,     0.0),
    (    0.0,  0.0203, 0.0,     0.0),
    (-np.pi/2,    0.0, 0.1500,  0.0),
    ( np.pi/2,    0.0, 0.0,     0.0),
    (-np.pi/2,    0.0, 0.0,     0.0)
]

# Joint limits 
JOINT_LIMITS = [
    np.deg2rad([-160, 160]),
    np.deg2rad([-225,  45]),
    np.deg2rad([ -45, 225]),
    np.deg2rad([-110, 170]),
    np.deg2rad([-100, 100]),
    np.deg2rad([-266, 266]),
]

# Mid‐points for IK initialization
Q_INIT = np.array([(low + high) / 2 for (low, high) in JOINT_LIMITS])

# Compute conservative reach bound (naive reach sphere, based on max arm radius)
REACH = sum(np.hypot(a, d) for (_, a, d, _) in DH_PARAMS)


class Robot:
    def __init__(self, dh_params):
        self.dh = dh_params

    def _dh_transform(self, alpha, a, d, theta):
        sa, ca = np.sin(alpha), np.cos(alpha)
        st, ct = np.sin(theta), np.cos(theta)
        return np.array([
            [   ct,   -st,    0,    a],
            [st*ca, ct*ca,  -sa, -sa*d],
            [st*sa, ct*sa,   ca,  ca*d],
            [    0,     0,    0,     1],
        ])

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i, (alpha, a, d, th_off) in enumerate(self.dh):
            T = T @ self._dh_transform(alpha, a, d, q[i] + th_off)
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        return pos, rot

    def _fk6(self, q):
        pos, rot = self.forward_kinematics(q)
        return np.hstack([pos, rot.as_rotvec()])

    def jacobian(self, q, eps=1e-6):
        f0 = self._fk6(q)
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros_like(q); dq[i] = eps
            J[:, i] = (self._fk6(q + dq) - self._fk6(q - dq)) / (2 * eps)
        return J

    def inverse_kinematics(self, target_pos, target_rot, q_init):
        lower = np.array([lim[0] for lim in JOINT_LIMITS])
        upper = np.array([lim[1] for lim in JOINT_LIMITS])
        tgt = np.hstack([target_pos, target_rot.as_rotvec()])

        def fun(q): return self._fk6(q) - tgt

        sol = least_squares(fun, q_init, bounds=(lower, upper),
                            xtol=TOL, ftol=TOL, verbose=0)
        return sol.x if sol.success else None


def compute_kappa(J):
    s = svd(J, compute_uv=False)
    return np.min(s) / np.max(s)


def main():
    robot = Robot(DH_PARAMS)

    # Sample reachable configurations uniformly in joint space
    reachable = []
    print(f"Sampling {N_REACHABLE} reachable configs …")
    pbar = tqdm(total=N_REACHABLE, desc="Reachable")
    while len(reachable) < N_REACHABLE:
        q = np.array([np.random.uniform(l, h) for l, h in JOINT_LIMITS])
        p, rot = robot.forward_kinematics(q)
        J = robot.jacobian(q)
        κ = compute_kappa(J)
        reachable.append((q, p, rot.as_quat(), κ)) # type: ignore
        pbar.update(1)
    pbar.close()

    # Determine conditioning thresholds
    kappas = np.array([r[3] for r in reachable])
    k_low  = np.percentile(kappas, PERCENTILE_LOW)
    k_high = np.percentile(kappas, PERCENTILE_HIGH)

    # Label reachable rows
    rows = []
    for q, p, quat, κ in reachable:
        lbl = 1 if κ < k_low else 3 if κ > k_high else 2
        rows.append(np.hstack([q, p, quat, lbl]))

    # Build KD-tree on reachable positions
    pos_cloud = np.array([r[1] for r in reachable])
    tree = cKDTree(pos_cloud)

    # Precompute KD-tree threshold
    EPS = NEIGHBOR_EPS_RATIO * REACH

    # Sample unreachable using KD-tree prefilter + occasional IK fallback
    print(f"Sampling {N_UNREACHABLE} unreachable poses …")
    pbar = tqdm(total=N_UNREACHABLE, desc="Unreachable")
    count = 0
    while count < N_UNREACHABLE:
        # position candidate
        pos = np.random.uniform(-REACH, REACH, size=3)
        dist, _ = tree.query(pos)

        if dist > EPS:
            # definitely unreachable
            q_raw = np.random.normal(size=4)
            q_raw /= np.linalg.norm(q_raw)
            rows.append(np.hstack([np.full(6, np.nan), pos, q_raw, 0]))
            count += 1
            pbar.update(1)
        else:
            # possible interior hole → fallback to IK check
            q_raw = np.random.normal(size=4)
            q_raw /= np.linalg.norm(q_raw)
            sol = robot.inverse_kinematics(pos, R.from_quat(q_raw), Q_INIT)
            if sol is None:
                rows.append(np.hstack([np.full(6, np.nan), pos, q_raw, 0]))
                count += 1
                pbar.update(1)
    pbar.close()

    # Save to CSV
    cols = [f"theta{i+1}" for i in range(6)] + ["x", "y", "z"] + \
           ["qx", "qy", "qz", "qw"] + ["label"]
    df = pd.DataFrame(rows, columns=cols)

    script_dir = Path(__file__).parent
    docs_dir   = script_dir.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_file   = docs_dir / "6DOF_classification_dataset.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved dataset with {len(df)} samples to {out_file}")


if __name__ == "__main__":
    main()