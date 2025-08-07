import torch.nn as nn
from config import HIDDEN_SIZES, DROPOUT

# July 20 3:13pm revision

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.act  = nn.GELU()
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.lin2(h)
        return self.norm(x + h)

class IKNet(nn.Module):
    """
    Feed-forward MLP mapping 2D task-space coords [x,y]
    to 3 joint-angle sin/cos pairs [sinθ1,cosθ1,…,sinθ3,cosθ3].
    """
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = 2                    # <— planar input

        # hidden layers with residual blocks
        for h in HIDDEN_SIZES:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h)) 
            layers.append(nn.GELU())
            layers.append(ResBlock(h))
            layers.append(nn.Dropout(DROPOUT))
            in_dim = h

        # Output layer: predicts [sinθ1,cosθ1,…,sinθ3,cosθ3] => 6 dims
        layers.append(nn.Linear(in_dim, 6))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B,2) batch of [x,y]
        Returns:
            torch.Tensor: shape (B,6) [sinθ1,cosθ1,…,sinθ3,cosθ3]
        """
        return self.net(x)
