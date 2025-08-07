import numpy as np

# Denavit Hartenberg Parameters describe robot geometry (a, alpha, d, theta_offset)
DH_PARAMS = [
    (0.0,  np.pi/2, 0.1,  0.0),  # Joint 1
    (0.5,  0.0,     0.0,  0.0),  # Joint 2
    (0.4,  0.0,     0.0,  0.0),  # Joint 3
    (0.0,  0.0,     0.0,  0.0),  # Joint 4 
]

# Joint limits: list of (min, max) in radians
JOINT_LIMITS = [
    (-np.pi,   np.pi),
    (-np.pi/2, np.pi/2),
    (-np.pi/2, np.pi/2),
    (-np.pi,   np.pi),
]

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Compute individual Denavit-Hartenberg transformation matrix.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,    -st*ca,  st*sa, a*ct],
        [st,     ct*ca, -ct*sa, a*st],
        [0.0,       sa,     ca,   d],
        [0.0,      0.0,    0.0, 1.0],
    ])

def forward_kinematics(thetas: np.ndarray) -> np.ndarray:
    """
    Compute end-effector pose (x, y, z, psi) for given 4 joint angles.

    Parameters:
        thetas: shape (4,), joint angles [θ1, θ2, θ3, θ4]

    Returns:
        pose: numpy array [x, y, z, psi]
    """
    T = np.eye(4)
    for (a, alpha, d, theta_off), theta in zip(DH_PARAMS, thetas):
        T = T @ dh_transform(a, alpha, d, theta + theta_off)

    # Position
    pos = T[:3, 3]

    # For wrist pitch, psi is rotation around the final joint axis.
    # End effector pitch angle about its own x axis in the world frame
    tool_axis = T[:3, 0]
    psi = np.arctan2(tool_axis[2], tool_axis[0])

    return np.hstack((pos, psi))

