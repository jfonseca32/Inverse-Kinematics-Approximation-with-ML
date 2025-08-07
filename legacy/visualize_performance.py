import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import robot 

# Load data & artifacts
df = pd.read_csv('test.csv')
true_poses = df[['x','y','z','psi']].values

scaler = joblib.load('scaler.pkl')
model  = joblib.load('ik_model.pkl')

# Predict angles and XYZ
X_scaled    = scaler.transform(true_poses)
pred_angles = model.predict(X_scaled)
pred_poses  = np.array([robot.forward_kinematics(ang) for ang in pred_angles])

# Take a sample
rng = np.random.RandomState(42)
subset_size = 50
idx = rng.choice(len(true_poses), subset_size, replace=False)
gt_xyz   = true_poses[idx, :3]
pred_xyz = pred_poses[idx, :3]

# Color map
colors = cm.viridis(np.linspace(0,1,subset_size))

# Plot
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')

for i, (gt, pr, c) in enumerate(zip(gt_xyz, pred_xyz, colors)):
    # true point
    ax.scatter(*gt, c=[c], marker='o', s=30)
    # predicted point
    ax.scatter(*pr, c=[c], marker='^', s=30)
    # connected by a line
    ax.plot([gt[0],pr[0]], [gt[1],pr[1]], [gt[2],pr[2]], c=c, alpha=0.6)

ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
ax.set_title('True (o) vs Predicted (^) End-Effector Positions')
plt.tight_layout()
plt.show()
