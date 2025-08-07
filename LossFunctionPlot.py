import numpy as np
import matplotlib.pyplot as plt

loss = np.load("ik_loss_curves.npz")
for k in loss.files:
    plt.plot(loss[k], label=k)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLPRegressor Training Loss (per cluster)")
plt.legend()
plt.tight_layout()
plt.show()