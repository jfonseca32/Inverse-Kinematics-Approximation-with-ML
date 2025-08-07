import torch.nn as nn
from config_classifier import HIDDEN_SIZE

class ReachabilityNet(nn.Module):
    """
    MLP classifier mapping 2D workspace coords [x,y]
    to one of 4 classes:
      0 = unreachable
      1 = reachable but near singularity
      2 = reachable & well-conditioned
      3 = reachable & high-manipulability
      Feed-forward MLP
    """
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),

        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),

        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),

        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),

        nn.Linear(hidden_size, 4)
)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2)
        Returns:
            logits: Tensor of shape (B, 4)
        """
        return self.net(x)
