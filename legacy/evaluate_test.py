import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import robot   

# Load test set
PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "docs" / "test.csv"
df_test = pd.read_csv(csv_path)

# True end‐effector poses
true_poses = df_test[['x','y','z','psi']].values 

# Load scaler & model 
PKL_DIR = PROJECT_ROOT / "docs"
scaler = joblib.load(PKL_DIR / "scaler.pkl")
model  = joblib.load(PKL_DIR / "ik_model.pkl")

# Predicted joint angles 
X_test_scaled = scaler.transform(true_poses)
y_pred_angles = model.predict(X_test_scaled)    

# Run forward kinematics on each prediction 
# robot.forward_kinematics expects a (4,) array and returns [x,y,z,psi]
pred_poses = np.array([
    robot.forward_kinematics(angles)
    for angles in y_pred_angles
])  

# Compute error metrics on positions 
# compare only x,y,z (end effector angle not considered yet)
true_pos = true_poses[:, :3]
pred_pos = pred_poses[:, :3]

pos_mse  = mean_squared_error(true_pos, pred_pos)
pos_rmse = np.sqrt(pos_mse)
pos_r2   = r2_score(true_pos, pred_pos, multioutput='uniform_average')

print(f"Position MSE: {pos_mse:.6f} (m^2)")
print(f"Position RMSE: {pos_rmse:.6f} m")
print(f"Position R^2: {pos_r2:.4f}")
