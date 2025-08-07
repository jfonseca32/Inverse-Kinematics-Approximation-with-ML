import os
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import joblib
from scipy.stats import qmc
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

class VersionChecker:
    """Utility to print version information for reproducibility."""

    @staticmethod
    def check():
        import sys
        import numpy as np
        import pandas as pd
        import pybullet
        import sklearn

        print("Python version:", sys.version)
        print("numpy version:", np.__version__)
        print("pandas version:", pd.__version__)
        print("pybullet version:", pybullet.__version__ if hasattr(pybullet, "__version__") else "n/a")
        print("scikit-learn version:", sklearn.__version__)

class DataValidator:
    """Utility to check for missing or infinite data in a DataFrame."""

    @staticmethod
    def check(df, name="DataFrame"):
        missing = df.isnull().sum().sum()
        infinite = (~np.isfinite(df.select_dtypes(include=[np.number]))).sum().sum()
        print(f"{name}: {missing} missing, {infinite} infinite values.")
        if missing == 0 and infinite == 0:
            print(f"{name} is clean.")
            return True
        else:
            print(f"{name} has issues! (Missing or infinite values present)")
            return False

class DatasetGenerator:
    """
    Generates a dataset of (joint angles, end-effector pose) pairs for an N-DOF manipulator.
    Sampling is done within the robot's joint limits (using Latin Hypercube Sampling).
    """
    def __init__(
        self,
        urdf_path: str,
        output_path: str = "ik_dataset.csv",
        n_samples: int = 10_000,
        batch_size: int = 1000,
        random_seed: int = 42,
        n_dof: int = 4,  # <<<<<<<<<<< KEY: Set your robot's DOF here!
    ):
        self.urdf_path = urdf_path
        self.output_path = output_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n_dof = n_dof

    def run(self, overwrite: bool = False) -> pd.DataFrame:
        """
        Generates or loads the dataset. Returns a DataFrame with columns:
        [theta1,...,thetaN, x,y,z, ...]
        """
        if os.path.exists(self.output_path) and not overwrite:
            print(f"Loading dataset from {self.output_path}")
            return pd.read_csv(self.output_path)

        np.random.seed(self.random_seed)
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        # Identify N revolute joint IDs and their limits
        joint_ids, joint_lows, joint_highs = [], [], []
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            if info[2] == p.JOINT_REVOLUTE and len(joint_ids) < self.n_dof:
                joint_ids.append(i)
                joint_lows.append(info[8])
                joint_highs.append(info[9])
        joint_lows, joint_highs = np.array(joint_lows), np.array(joint_highs)
        assert len(joint_ids) == self.n_dof, f"Robot model must have at least {self.n_dof} revolute joints."

        # Latin Hypercube Sampling for joint angles
        sampler = qmc.LatinHypercube(d=self.n_dof)
        unit = sampler.random(n=self.n_samples)
        sampled_thetas = qmc.scale(unit, joint_lows, joint_highs)

        records = []
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            for idx in trange(start, end, desc=f"Generating {start}-{end}"):
                angles = sampled_thetas[idx]
                # Set joint positions
                for jid, ang in zip(joint_ids, angles):
                    p.resetJointState(robot, jid, ang)
                link = p.getLinkState(robot, joint_ids[-1], computeForwardKinematics=True)
                pos, orn = link[4], link[5]  # [x,y,z], quaternion [qx,qy,qz,qw]
                records.append([*angles, *pos, *orn])

        p.disconnect()

        # DataFrame columns
        angle_cols = [f"theta{i+1}" for i in range(self.n_dof)]
        pose_cols = ["x", "y", "z", "qx", "qy", "qz", "qw"]  # Remove quat columns if unnecessary
        df = pd.DataFrame(records, columns=angle_cols + pose_cols)
        df.to_csv(self.output_path, index=False)
        print(f"Saved dataset to {self.output_path} ({len(df)} rows)")
        return df

class IKDataEDA:
    """
    Exploratory Data Analysis for IK Datasets.
      - Workspace scatter plots
      - Joint angle histograms
      - Pairwise (angle/pose) scatter matrices
      - Cluster-colored workspace for multi-solution visualization
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_workspace(self, sample_size=5000, color=None, elev=30, azim=-60):
        """3D scatter of end-effector positions (optionally colored by cluster/label)."""
        if sample_size < len(self.df):
            sample = self.df.sample(sample_size)
        else:
            sample = self.df
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        if color is None:
            ax.scatter(sample['x'], sample['y'], sample['z'], s=3, alpha=0.5)
        else:
            c = sample[color] if isinstance(color, str) else color
            p = ax.scatter(sample['x'], sample['y'], sample['z'], c=c, cmap='viridis', s=5, alpha=0.8)
            fig.colorbar(p, ax=ax, label="Cluster ID")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        plt.title("End-Effector Workspace")
        plt.tight_layout()
        plt.show()

    def plot_joint_histograms(self, bins=60):
        """Histograms of each joint angle (auto grid for DOF)."""
        theta_cols = [col for col in self.df.columns if col.startswith("theta")]
        n = len(theta_cols)
        ncols = 3 if n >= 3 else n
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)  # Flatten in case n < nrows*ncols
        for i, col in enumerate(theta_cols):
            ax = axes[i]
            ax.hist(self.df[col], bins=bins, color="C0", alpha=0.8)
            ax.set_title(f"{col} Distribution")
            ax.set_xlabel("Angle (rad)")
            ax.set_ylabel("Count")
        # Hide any unused axes
        for i in range(len(theta_cols), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_joint_pairplot(self, sample_size=3000):
        """Pairplot (scatter matrix) for joint angles to see correlations/multimodality."""
        theta_cols = [col for col in self.df.columns if col.startswith("theta")]
        if sample_size < len(self.df):
            sample = self.df.sample(sample_size)
        else:
            sample = self.df
        sns.pairplot(sample[theta_cols], diag_kind="hist", plot_kws={'alpha':0.4, 's':10})
        plt.suptitle("Pairwise Joint Angle Scatter Matrix", y=1.02)
        plt.show()

    def plot_orientation_histograms(self, bins=60):
        """Histogram of end-effector Euler angles (computed from quaternions)."""
        quats = self.df[["qx","qy","qz","qw"]].to_numpy()
        eulers = R.from_quat(quats).as_euler('xyz', degrees=True)
        labels = ['roll', 'pitch', 'yaw']
        fig, axes = plt.subplots(1, 3, figsize=(18,4))
        for i, ax in enumerate(axes):
            ax.hist(eulers[:, i], bins=bins, alpha=0.7, color="C1")
            ax.set_title(f"{labels[i].capitalize()} Distribution")
            ax.set_xlabel("Degrees")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_workspace_by_cluster(self, clusters, sample_size=5000, elev=30, azim=-60):
        """
        3D workspace plot colored by cluster assignments (e.g., elbow up/down).
        Useful for visualizing multi-solution regions.
        clusters: np.array (N,) or Series, or column name in self.df
        """
        if isinstance(clusters, str):
            color = clusters
            self.plot_workspace(sample_size=sample_size, color=color, elev=elev, azim=azim)
        else:
            if sample_size < len(self.df):
                idx = np.random.choice(len(self.df), sample_size, replace=False)
                sample = self.df.iloc[idx].copy()
                c = np.array(clusters)[idx]
            else:
                sample = self.df
                c = clusters
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(sample['x'], sample['y'], sample['z'], c=c, cmap='viridis', s=5, alpha=0.8)
            fig.colorbar(p, ax=ax, label="Cluster ID")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            plt.title("End-Effector Workspace by Cluster")
            plt.tight_layout()
            plt.show()

class IKClusterer:
    """
    KMeans clustering on joint angle vectors.
    Used to partition the dataset into families of solutions (e.g., 'elbow up/down').
    """
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

    def fit(self, joint_angles: np.ndarray):
        """
        Fit KMeans to joint angles.
        Args:
            joint_angles: shape (N, 4) ndarray of [theta1, ..., theta6]
        Returns:
            cluster_labels: array of shape (N,) with cluster assignments (0, 1, ...)
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(joint_angles)
        return cluster_labels

    def predict(self, joint_angles: np.ndarray):
        """
        Assign cluster labels for new joint angles.
        """
        if self.kmeans is None:
            raise RuntimeError("You must fit the clusterer first!")
        return self.kmeans.predict(joint_angles)

    def save(self, filename):
        """
        Save the KMeans model for later use.
        """
        joblib.dump(self.kmeans, filename)

    def load(self, filename):
        """
        Load a saved KMeans model.
        """
        self.kmeans = joblib.load(filename)

class PerClusterRegressor:
    """
    Trains and manages one regressor per cluster label.
    Each regressor predicts joint angles from normalized pose inputs,
    using only samples assigned to its cluster.
    """
    def __init__(self, base_regressor=None):
        """
        base_regressor: scikit-learn regressor instance
        (e.g., MLPRegressor, GradientBoostingRegressor).
        If None, defaults to MLPRegressor.
        """
        if base_regressor is None:
            self.base_regressor = MLPRegressor(
                hidden_layer_sizes=(256, 256, 256),
                max_iter=500,
                random_state=42
            )
        else:
            self.base_regressor = base_regressor
        self.cluster_models = {}  # cluster_label -> regressor

    def fit(self, X, Y, clusters):
        """
        Trains a regressor per cluster.
        X: (N, D) pose inputs (normalized)
        Y: (N, 4) joint angle outputs (can be normalized or not, but must be consistent at inference)
        clusters: (N,) cluster label for each sample
        """
        self.unique_clusters = np.unique(clusters)
        for c in self.unique_clusters:
            idx = np.where(clusters == c)[0]
            X_c, Y_c = X[idx], Y[idx]
            model = clone(self.base_regressor)
            model.fit(X_c, Y_c)
            self.cluster_models[c] = model

    def predict(self, X, fk_func=None, poses=None):
        """
        Predicts joint angles for each input sample.
        If fk_func and poses are provided: runs all cluster regressors and selects the output
        whose forward kinematics is closest to the input pose (as in the paper).
        Otherwise, user must provide a method to map X to clusters.
        Args:
            X: (M, D) pose inputs (normalized)
            fk_func: callable (joint angles) -> pose (x,y,z or x,y,z,qx,qy,qz,qw)
            poses: (M, D) the original pose for each sample (for selecting best match)
        Returns:
            predictions: (M, 4) array of predicted joint angles
        """
        M = X.shape[0]
        n_clusters = len(self.unique_clusters)
        if fk_func is not None and poses is not None:
            # Predict with all regressors, choose best via FK error
            preds = np.zeros((M, n_clusters, 4))
            for i, c in enumerate(self.unique_clusters):
                preds[:, i, :] = self.cluster_models[c].predict(X)
            # For each sample, select the regressor whose FK is closest to the input pose
            best_preds = []
            for j in range(M):
                fk_errors = [np.linalg.norm(fk_func(preds[j, i])[:3] - poses[j][:3]) for i in range(n_clusters)]
                best_idx = np.argmin(fk_errors)
                best_preds.append(preds[j, best_idx])
            return np.vstack(best_preds)
        else:
            raise NotImplementedError("For multi-solution problems, must supply fk_func and original poses for "
                                      "proper cluster assignment at inference.")

    def predict_per_cluster(self, X, cluster_labels):
        """
        Predicts joint angles using only the regressor for the provided cluster label.
        Use only if you know which cluster a pose belongs to (not typical for real-world inference).
        """
        preds = []
        for x, c in zip(X, cluster_labels):
            preds.append(self.cluster_models[c].predict(x.reshape(1,-1))[0])
        return np.vstack(preds)

def clustered_ik_predict(
    regressors,    # List of trained regressors, one per cluster
    pose,          # (D,) or (1,D) normalized pose input
    fk_func,       # Forward kinematics: joint angles -> pose
    input_pose,    # The desired pose (unnormalized), to compare FK output against
):
    """
    Predict joint angles for a single input pose by evaluating all cluster regressors
    and choosing the output whose FK is closest to the desired pose.
    """
    min_error = float('inf')
    best_pred = None
    for reg in regressors:
        pred_angles = reg.predict(pose.reshape(1,-1))[0]
        pred_pose = fk_func(pred_angles)
        error = np.linalg.norm(np.array(pred_pose[:3]) - np.array(input_pose[:3]))
        if error < min_error:
            min_error = error
            best_pred = pred_angles
    return best_pred

def batch_clustered_ik_predict(
    regressors,
    poses,            # (N, D) normalized poses
    fk_func,
    input_poses       # (N, D) desired (unnormalized) poses
):
    """
    Batch version: predicts for multiple poses (N, D).
    """
    preds = []
    for pose, input_pose in zip(poses, input_poses):
        best_pred = clustered_ik_predict(regressors, pose, fk_func, input_pose)
        preds.append(best_pred)
    return np.vstack(preds)

class IKEvaluator:
    """
    Evaluation utilities for inverse kinematics regressors:
      - Joint-space MAE
      - End-effector position error via FK
    """

    def __init__(self, fk_func):
        """
        fk_func: callable, joint angles -> end-effector pose (should return at least [x, y, z])
        """
        self.fk_func = fk_func

    def joint_mae(self, y_true, y_pred):
        """
        Mean Absolute Error in joint space (angles, per sample averaged over all joints)
        """
        return mean_absolute_error(y_true, y_pred)

    def ee_position_error(self, y_true, y_pred):
        """
        Mean Euclidean error in workspace (end-effector positions).
        Both y_true and y_pred are joint angles; FK is applied.
        Returns mean error and all errors.
        """
        pos_true = np.array([self.fk_func(ang)[:3] for ang in y_true])
        pos_pred = np.array([self.fk_func(ang)[:3] for ang in y_pred])
        dists = np.linalg.norm(pos_pred - pos_true, axis=1)
        return dists.mean(), dists

# Clone of function in Jupyter Notebook to enable use in other files (like VisualizeResults)
class FastFK:
    """
    Utility class for forward kinematics (FK) evaluation using PyBullet.
    Returns the end-effector pose for a given joint configuration.
    """
    def __init__(self, urdf_path="simple_4dof.urdf", n_dof=4):
        """
        Initializes the PyBullet simulation, loads the robot from URDF,
        and identifies the revolute joints to use for FK.

        Args:
            urdf_path (str): Path to your URDF file.
            n_dof (int): Number of revolute joints (must match robot DOF).
        """
        import pybullet as p
        import pybullet_data
        self.urdf_path = urdf_path
        self.n_dof = n_dof
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.joint_ids = [
            i for i in range(p.getNumJoints(self.robot))
            if p.getJointInfo(self.robot, i)[2] == p.JOINT_REVOLUTE
        ][:self.n_dof]
        assert len(self.joint_ids) == self.n_dof, f"Robot model must have {self.n_dof} revolute joints."

    def __call__(self, joint_angles):
        """
        Computes the end-effector pose for a given set of joint angles.

        Args:
            joint_angles (array-like): Joint angles in radians.

        Returns:
            np.ndarray: [x, y, z, qx, qy, qz, qw] pose of the end effector.
        """
        import pybullet as p
        for jid, ang in zip(self.joint_ids, joint_angles):
            p.resetJointState(self.robot, jid, ang)
        link_state = p.getLinkState(self.robot, self.joint_ids[-1], computeForwardKinematics=True)
        pos, orn = link_state[4], link_state[5]
        return np.array([*pos, *orn])

    def disconnect(self):
        """
        Disconnects from the PyBullet simulation.
        """
        import pybullet as p
        p.disconnect(self.client)
