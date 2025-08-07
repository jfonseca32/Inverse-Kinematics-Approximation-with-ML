import torch.nn as nn
from config_classifier_6DOF import HIDDEN_SIZE

class ReachabilityNet6D(nn.Module):
    """
    MLP classifier mapping 7D pose coords [x, y, z, qx, qy, qz, qw]
    to one of 4 classes:
      0 = unreachable
      1 = reachable but near singularity
      2 = reachable & well-conditioned
      3 = reachable & high-manipulability
    Uses a feed-forward MLP.
    """
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.ReLU(),

            # Linear + rectified lienar for nonlinearity
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # final logits → 4 classes
            nn.Linear(hidden_size, 4)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 7)  where each row is [x,y,z,qx,qy,qz,qw]
        Returns:
            logits: Tensor of shape (B, 4)
        """
        return self.net(x)
