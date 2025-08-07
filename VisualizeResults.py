import numpy as np
import matplotlib.pyplot as plt
from Methods import FastFK

# 1. Load the results
data = np.load("ik_results_for_viz.npz")
Y_true = data['Y_true']
Y_pred = data['Y_pred']

# 2. (Re)instantiate your FastFK class to get a fresh PyBullet connection
fast_fk = FastFK(urdf_path="simple_4dof.urdf")

# 3. Plot function
def plot_ee_positions(y_true, y_pred, fk_func, n_samples=100, joint_labels=None):
    idxs = np.random.choice(len(y_true), size=n_samples, replace=False)
    true_xyz = np.array([fk_func(j)[:3] for j in y_true[idxs]])
    pred_xyz = np.array([fk_func(j)[:3] for j in y_pred[idxs]])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_xyz[:,0], true_xyz[:,1], true_xyz[:,2], c='g', marker='o', label='Ground Truth EE')
    ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2], c='r', marker='^', label='Predicted EE')
    for i in range(n_samples):
        ax.plot([true_xyz[i,0], pred_xyz[i,0]],
                [true_xyz[i,1], pred_xyz[i,1]],
                [true_xyz[i,2], pred_xyz[i,2]], 'k--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'EE: Green=Truth, Red=Pred, {n_samples} Random Samples')
    plt.tight_layout()
    plt.show()

# 4. Use the function
plot_ee_positions(Y_true, Y_pred, fk_func=fast_fk, n_samples=100)

# 5. Disconnect when done
fast_fk.disconnect()