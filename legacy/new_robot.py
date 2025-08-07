import numpy as np

# Denavit-Hartenberg parameters for a planar 3R robot
# Each entry is a dict with:
#   a: link length
#   alpha: twist angle (0 for planar)
#   d: link offset (0 for planar)
#   theta_offset: constant joint angle offset
DH_PARAMS = [
    { 'a': 0.25, 'alpha': 0.0, 'd': 0.0, 'theta_offset': 0.0 },  # joint 1
    { 'a': 0.25, 'alpha': 0.0, 'd': 0.0, 'theta_offset': 0.0 },  # joint 2
    { 'a': 0.25, 'alpha': 0.0, 'd': 0.0, 'theta_offset': 0.0 },  # joint 3
]

# Joint limits for a planar 3R arm (radians)
JOINT_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]


class Robot:
    """
    Planar 3R robot model using Denavit-Hartenberg parameters.

    Attributes:
        dh_params: list of DH parameter dicts
        n_joints: number of rotational joints (3)
    """
    def __init__(self, dh_params=DH_PARAMS):
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    def _dh_transform(self, a, alpha, d, theta):
        """
        Compute DH transform matrix for planar joints (alpha=0, d=0).

        Returns a 4x4 numpy.ndarray.
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ ct, -st * ca,  st * sa, a * ct],
            [ st,  ct * ca, -ct * sa, a * st],
            [  0,       sa,       ca,      d],
            [  0,        0,        0,      1]
        ])

    def forward_kinematics(self, q):
        """
        Computes planar end-effector (x, y) for given joint angles.

        Args:
            q (array-like): [theta1, theta2, theta3]

        Returns:
            np.ndarray: shape (2,) [x, y]
        """
        T = np.eye(4)
        for i, param in enumerate(self.dh_params):
            a = param['a']
            alpha = param['alpha']
            d = param['d']
            theta = q[i] + param.get('theta_offset', 0.0)
            T = T @ self._dh_transform(a, alpha, d, theta)
        # Extract planar x, y
        return T[0:2, 3]

    def jacobian(self, q):
        """
        Compute the 2×3 Jacobian matrix d[x,y]/d[θ1,θ2,θ3].

        Args:
            q (array-like): [theta1, theta2, theta3]

        Returns:
            np.ndarray: shape (2, 3) Jacobian matrix
        """
        T = np.eye(4)
        origins = [T[:3, 3]]
        z_axes = [T[:3, 2]]

        # forward pass to collect each joint origin and z-axis
        for i, param in enumerate(self.dh_params):
            a = param['a']
            alpha = param['alpha']
            d = param['d']
            theta = q[i] + param.get('theta_offset', 0.0)
            T = T @ self._dh_transform(a, alpha, d, theta)
            origins.append(T[:3, 3])
            z_axes.append(T[:3, 2])

        # full cross-product Jacobian (3×3)
        p_end = origins[-1]
        J_full = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            J_full[:, i] = np.cross(z_axes[i], p_end - origins[i])

        # return only planar part (dx/dtheta, dy/dtheta)
        return J_full[0:2, :]
