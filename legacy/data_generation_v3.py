from new_robot import Robot, DH_PARAMS, JOINT_LIMITS
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import brent


def sample_workspace(N):
    """
    Randomly samples accessible joint space of robot within JOINT_LIMITS.
    Uses forward kinematics to transfer random samples to task-space [x, y, z].

    Args:
        N (int): desired number of samples

    Returns:
        points (np.ndarray): shape (N,3) array of task-space coordinates ([x y z])
        q_samples (np.ndarray): shape (N,4) array of original joint angles [θ1..θ4]
    """
    robot = Robot()
    num_joints = robot.n_joints
    q_samples = np.zeros((N, num_joints))

    # sample joint angles uniformly
    for i, (low, high) in enumerate(JOINT_LIMITS):
        q_samples[:, i] = np.random.uniform(low, high, size=N)

    # forward kinematics to task-space
    points = np.array([robot.forward_kinematics(q) for q in q_samples])
    return points, q_samples


def ik_2joint(robot, target, q3, q12_init, tol, max_iter):
    """
    Solve for θ1, θ2 given fixed θ3 via Newton-Raphson on the 2×2 subproblem.
    """
    q12 = np.array(q12_init, dtype=float)
    for _ in range(max_iter):
        q_full = np.array([q12[0], q12[1], q3], dtype=float)
        p_cur = robot.forward_kinematics(q_full)    # shape (2,)
        err = p_cur - target
        if np.linalg.norm(err) < tol:
            # check joint limits
            for i in range(2):
                low, high = JOINT_LIMITS[i]
                if not (low <= q12[i] <= high):
                    return None
            return q12
        J = robot.jacobian(q_full)                 # shape (2,3)
        J2 = J[:, :2]                              # shape (2,2)
        try:
            delta = np.linalg.solve(J2, err)
        except np.linalg.LinAlgError:
            return None
        q12 = q12 - delta
    return None

def w_of_q3(robot, target, q3, q12_init, tol, max_iter):
    """
    Given θ3, solve for θ1,θ2 then compute manipulability w.
    Returns w (>= 0) or -inf if IK failed.
    """
    q12 = ik_2joint(robot, target, q3, q12_init, tol, max_iter)
    if q12 is None:
        return -np.inf
    q_full = np.array([q12[0], q12[1], q3], dtype=float)
    J = robot.jacobian(q_full)                  # shape (2,3)
    JJt = J @ J.T                               # shape (2,2)
    det_val = np.linalg.det(JJt)
    return np.sqrt(det_val) if det_val > 0 else 0.0

def find_best_solution(robot, target, q_init, tol, max_iter):
    """
    Find θ3 in [limit] that maximizes w, using Brent on the 1-D function.
    Returns (q_best (3,), w_best) or (None, None) if no valid IK.
    """
    q3_min, q3_max = JOINT_LIMITS[2]
    # negative w for minimization
    func = lambda q3: -w_of_q3(robot, target, q3, q_init[:2], tol, max_iter)
    # run Brent search
    try:
        q3_best = brent(func, brack=(q3_min, q3_max))
    except Exception:
        return None, None

    w_best = w_of_q3(robot, target, q3_best, q_init[:2], tol, max_iter)
    if w_best < 0:
        return None, None

    # recover θ1,θ2 for this best q3
    q12_best = ik_2joint(robot, target, q3_best, q_init[:2], tol, max_iter)
    if q12_best is None:
        return None, None

    q_best = np.array([q12_best[0], q12_best[1], q3_best], dtype=float)
    return q_best, w_best

if __name__ == '__main__':
    # Desired number of valid samples
    N_PTS      = 50000
    TOL        = 1e-6
    MAX_ITER   = 10
    W_MIN      = 0.005  # minimum manipulability threshold

    # output path
    script_dir = Path(__file__).parent
    docs_dir   = script_dir.parent / 'docs'
    docs_dir.mkdir(exist_ok=True)
    out_file   = docs_dir / 'manipulability_dataset.csv'

    robot = Robot()
    rows  = []
    pbar  = tqdm(total=N_PTS, desc="Building dataset")

    while len(rows) < N_PTS:
        to_sample    = N_PTS - len(rows)
        points, qs   = sample_workspace(to_sample)
        for p, q0 in zip(points, qs):
            q_best, w_best = find_best_solution(robot, p, q0, TOL, MAX_ITER)
            if q_best is not None and w_best >= W_MIN:
                rows.append([*p, *q_best, w_best])
                pbar.update(1)
                if len(rows) >= N_PTS:
                    break

    pbar.close()

    df = pd.DataFrame(rows, columns=['x', 'y', 'θ1', 'θ2', 'θ3', 'w_max'])
    df.to_csv(out_file, index=False)