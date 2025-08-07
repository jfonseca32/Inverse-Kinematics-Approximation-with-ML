from new_robot import Robot, DH_PARAMS, JOINT_LIMITS
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

def sample_workspace(N):
    """
    Randomly samples accesible joint space of robot within Robot variable JOINT_LIMITS.
    Uses forward kinematics to transfer random samples to task-space [x y z].
    
    Args:
        N (int): desired number of samples

    Returns:
        points (np.ndarray): shape (N,3) array of task-space coordinates ([x y z])
        q_samples (np.ndarray): shape (N,4) array of original joint angles [θ1..θ4] used to generate xyz
    """
    robot = Robot()
    num_joints = robot.n_joints
    q_samples = np.zeros((N, num_joints))

    for i, (low, high) in enumerate(JOINT_LIMITS):
        q_samples[:, i] = np.random.uniform(low, high, size=N)

    # Compute forward kinematics for each sample
    points = np.array([robot.forward_kinematics(q) for q in q_samples])
    return points, q_samples

def newton_ik(robot, target, q4, q123_init, tol, max_iter):
    """
    Solve for joints 1-3 via Newton-Raphson for a fixed q4, starting from q123_init.

    Args:
        robot (Robot): Robot instance 
        target (array like): target [x y z] end effector position
        q4 (float): fixed angle for joint 4
        q123_init (array like): initial guess for [θ1, θ2, θ3]
        tol (float): convergence tolerance on position error
        max_iter (int): maximum number of Newton iterations

    Returns:
        np.ndarray or None: shape (3,) array of [θ1, θ2, θ3] if successful, otherwise None
    """
    # Initial guess for q1-3
    q123 = np.array(q123_init, dtype=float)
    for _ in range(max_iter):
        # full joint vector [q1, q2, q3, q4]
        q_full = np.concatenate([q123, [q4]])
        # current end-effector position
        p_cur = robot.forward_kinematics(q_full)
        # position error
        err = p_cur - target
        if np.linalg.norm(err) < tol:
            # joint limits check for q1-3
            for i in range(3):
                low, high = JOINT_LIMITS[i]
                if not (low <= q123[i] <= high):
                    return None
            return q123
        # compute jacobian and take first 3 columns
        J = robot.jacobian(q_full)
        J3 = J[:, :3]
        try:
            delta = np.linalg.solve(J3, err)
        except np.linalg.LinAlgError:
            return None
        q123 = q123 - delta
    return None


def manipulability(robot, q):
    """
    Compute the manipulability index, w, for a full joint vector.

    w = sqrt(det(J * J^T))

    Args:
        robot (Robot): Robot instance
        q (array like): full joint vector

    Returns:
        float: manipulability scalar (>=0)
    """
    J = robot.jacobian(q)
    JJt = J @ J.T
    det_val = np.linalg.det(JJt)
    # avoid negative due to round-off
    w = np.sqrt(det_val) if det_val > 0 else 0.0
    return w

def sweep_redundant(robot, target, q4_limits, q123_init, M, tol, max_iter):
    """
    Sweep the redundant joint q4 across M samples, using q123_init as the warm start for newton_ik.

    Args:
        robot (Robot): Robot instance
        target (array like): target [x, y, z]
        q4_limits (tuple): (min_q4, max_q4)
        q123_init (array like): initial guess for [q1,q2,q3]
        M (int): number of q4 samples
        tol (float): position tolerance
        max_iter (int): max IK iterations

    Returns:
        list of (q_full, w) tuples for valid solutions
    """
    results = []
    q4_vals = np.linspace(q4_limits[0], q4_limits[1], M)
    # warm start for q1-3
    q123_guess = np.array(q123_init, dtype=float)
    for q4 in q4_vals:
        q123 = newton_ik(robot, target, q4, q123_guess, tol, max_iter)
        if q123 is not None:
            q_full = np.concatenate([q123, [q4]])
            # update warm start for next iteration
            q123_guess = q123
            # compute manipulability
            J = robot.jacobian(q_full)
            w = np.sqrt(np.linalg.det(J @ J.T)) if np.linalg.det(J @ J.T) > 0 else 0.0
            results.append((q_full, w))
    return results

def find_best_solution(robot, target, q4_limits, q123_init, M, tol, max_iter):
    """
    Find the IK solution that maximizes manipulability for given xyz

    Returns:
        tuple: (q_best, w_best) manipulability-optimal joint angles and corresponding w-value
    """
    candidates = sweep_redundant(robot, target, q4_limits, q123_init, M, tol, max_iter)
    if not candidates:
        return None, None
    q_best, w_best = max(candidates, key=lambda cw: cw[1])
    return q_best, w_best

# Command line usage: python src/new_data_generation.py
if __name__ == '__main__':
    # default params
    N_PTS = 50000
    M_REDUNDANT = 100
    TOL = 1e-6
    MAX_ITER = 10

    # output path (docs folder)
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / 'docs'
    docs_dir.mkdir(exist_ok=True)
    out_file = docs_dir / 'manipulability_dataset.csv'

    robot = Robot()
    points, q_samples = sample_workspace(N_PTS)
    rows = []
    for p, q0 in tqdm(zip(points, q_samples),
                      total=len(points),
                      desc="Building dataset"):

        q_best, w_best = find_best_solution(
            robot, p, JOINT_LIMITS[3], q0[:3], M_REDUNDANT, TOL, MAX_ITER
        )
        if q_best is not None:
            rows.append([*p, *q_best, w_best])
    df = pd.DataFrame(rows, columns=['x', 'y', 'z', 'θ1', 'θ2', 'θ3', 'θ4', 'w_max'])
    df.to_csv(str(out_file), index=False)
