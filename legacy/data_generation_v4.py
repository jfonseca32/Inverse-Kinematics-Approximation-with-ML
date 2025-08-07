import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import brent

from new_robot import Robot, JOINT_LIMITS, DH_PARAMS

# --- Hyperparameters ---
N_REACHABLE    = 100000
N_UNREACHABLE  = 40000
TOL            = 1e-6
MAX_ITER       = 10
PERCENTILE_LOW = 10    # bottom 10% ⇒ near-singularity
PERCENTILE_HIGH= 90    # top 10% ⇒ high-conditioning

# Compute maximum reach radius
REACH = sum(p['a'] for p in DH_PARAMS)

# Initial guess for IK (midpoint of joint limits)
Q_INIT = np.array([(low+high)/2 for (low, high) in JOINT_LIMITS], dtype=float)


def ik_2joint(robot, target, q3, q12_init, tol, max_iter):
    """
    Solve for θ1, θ2 given fixed θ3 via Newton-Raphson.
    """
    q12 = np.array(q12_init, dtype=float)
    for _ in range(max_iter):
        q_full = np.array([q12[0], q12[1], q3], dtype=float)
        p_cur = robot.forward_kinematics(q_full)
        err = p_cur - target
        if np.linalg.norm(err) < tol:
            for i in range(2):
                low, high = JOINT_LIMITS[i]
                if not (low <= q12[i] <= high):
                    return None
            return q12
        J = robot.jacobian(q_full)[:, :2]  # take first 2 columns
        try:
            delta = np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            return None
        q12 -= delta
    return None


def metric_of_q3(robot, target, q3, q12_init, tol, max_iter):
    """
    Solve IK for given q3 and return inverse condition number σ_min/σ_max.
    Returns -inf if IK fails.
    """
    q12 = ik_2joint(robot, target, q3, q12_init, tol, max_iter)
    if q12 is None:
        return -np.inf
    q_full = np.array([q12[0], q12[1], q3], dtype=float)
    J = robot.jacobian(q_full)
    JJt = J @ J.T
    lams = np.linalg.eigvalsh(JJt)      # eigenvalues λ = σ^2
    s_min = np.sqrt(max(lams[0], 0.0))
    s_max = np.sqrt(max(lams[1], 0.0))
    return s_min / (s_max + 1e-12)


def find_best_solution(robot, target, q_init, tol, max_iter):
    """
    Find q3 ∈ limits that maximizes inverse condition number via Brent.
    Returns (q_best (3,), kappa_inv_best) or (None, None).
    """
    q3_min, q3_max = JOINT_LIMITS[2]
    func = lambda q3: -metric_of_q3(robot, target, q3, q_init[:2], tol, max_iter)
    try:
        q3_best = brent(func, brack=(q3_min, q3_max))
    except Exception:
        return None, None

    kappa_inv = metric_of_q3(robot, target, q3_best, q_init[:2], tol, max_iter)
    if kappa_inv < 0:
        return None, None

    q12_best = ik_2joint(robot, target, q3_best, q_init[:2], tol, max_iter)
    if q12_best is None:
        return None, None

    q_best = np.array([q12_best[0], q12_best[1], q3_best], dtype=float)
    return q_best, kappa_inv


if __name__ == '__main__':
    # prepare output path
    script_dir = Path(__file__).parent
    docs_dir   = script_dir.parent / 'docs'
    docs_dir.mkdir(exist_ok=True)
    out_file   = docs_dir / 'classification_dataset.csv'

    robot = Robot()

    # 1) Generate reachable samples and collect metrics ---
    reachable_data = []
    pbar = tqdm(total=N_REACHABLE, desc="Sampling reachable")
    while len(reachable_data) < N_REACHABLE:
        # sample random joint angles
        q_rand = np.array([
            np.random.uniform(low, high)
            for (low, high) in JOINT_LIMITS
        ], dtype=float)
        target = robot.forward_kinematics(q_rand)  # [x,y]
        q_best, kappa_inv = find_best_solution(robot, target, Q_INIT, TOL, MAX_ITER)
        if q_best is not None:
            reachable_data.append((target[0], target[1], kappa_inv))
            pbar.update(1)
    pbar.close()

    # extract kappa_inv values to compute thresholds
    kappa_vals = np.array([val for (_, _, val) in reachable_data])
    kappa_low  = np.percentile(kappa_vals, PERCENTILE_LOW)
    kappa_high = np.percentile(kappa_vals, PERCENTILE_HIGH)

    # 2) Label reachable points into three classes ---
    rows = []
    for x, y, kappa_inv in reachable_data:
        if kappa_inv < kappa_low:
            label = 1  # near-singularity
        elif kappa_inv > kappa_high:
            label = 3  # high-conditioning
        else:
            label = 2  # well-conditioned
        rows.append([x, y, label])

    # 3) Generate unreachable samples ---
    pbar = tqdm(total=N_UNREACHABLE, desc="Sampling unreachable")
    count_unreach = 0
    while count_unreach < N_UNREACHABLE:
        x = np.random.uniform(-REACH, REACH)
        y = np.random.uniform(-REACH, REACH)
        q_best, _ = find_best_solution(robot, np.array([x, y]), Q_INIT, TOL, MAX_ITER)
        if q_best is None:
            rows.append([x, y, 0])  # unreachable
            count_unreach += 1
            pbar.update(1)
    pbar.close()

    # 4) Save to CSV ---
    df = pd.DataFrame(rows, columns=['x', 'y', 'label'])
    df.to_csv(out_file, index=False)
    print(f"Saved classification dataset ({len(rows)} samples) to {out_file}")
