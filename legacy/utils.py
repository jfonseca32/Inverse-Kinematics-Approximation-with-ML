import torch
import os
from new_robot import DH_PARAMS

def fk_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Batch-capable forward kinematics in PyTorch based on DH_PARAMS
    for a planar 3R robot.

    Args:
        q: [..., 3] tensor of joint angles [θ1, θ2, θ3]

    Returns:
        [..., 2] tensor of end-effector positions [x, y]
    """
    orig_shape = q.shape
    # flatten batch dimensions, keep last dim = 3
    q_flat = q.view(-1, orig_shape[-1])  # [B,3]
    B, _ = q_flat.shape
    device = q_flat.device
    dtype = q_flat.dtype

    # start with identity transforms
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]

    for i, param in enumerate(DH_PARAMS):
        a = torch.tensor(param['a'], device=device, dtype=dtype)
        alpha = torch.tensor(param['alpha'], device=device, dtype=dtype)
        d = torch.tensor(param['d'], device=device, dtype=dtype)
        theta_off = torch.tensor(param.get('theta_offset', 0.0), device=device, dtype=dtype)

        theta = q_flat[:, i] + theta_off  # [B]

        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        ct = torch.cos(theta)
        st = torch.sin(theta)

        zeros = torch.zeros_like(theta)
        ones  = torch.ones_like(theta)

        row0 = torch.stack([ct, -st * ca,  st * sa, a * ct], dim=1)  # [B,4]
        row1 = torch.stack([st,  ct * ca, -ct * sa, a * st], dim=1)
        row2 = torch.stack([zeros, sa * ones, ca * ones, d * ones], dim=1)
        row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)
        Ti   = torch.stack([row0, row1, row2, row3], dim=1)         # [B,4,4]

        T = T.bmm(Ti)  # batch matrix multiply

    # extract x,y from homogeneous transform
    pos = T[..., :2, 3]  # [B,2]
    return pos.view(*orig_shape[:-1], 2)


def jacobian_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the geometric Jacobian J for a batch of joint vectors using autograd,
    for a planar 3R robot.

    Args:
        q: [B,3] tensor of joint angles [θ1, θ2, θ3]

    Returns:
        [B,2,3] tensor of Jacobians mapping joint velocities to planar linear velocity
    """
    orig_shape = q.shape
    # flatten and enable gradients
    q_flat = q.clone().detach().requires_grad_(True).view(-1, orig_shape[-1])  # [B,3]
    J_list = []
    for qi in q_flat:
        # qi: [3]
        def fk_single(x):
            return fk_torch(x.unsqueeze(0))[0]  # returns [2] position
        Ji = torch.autograd.functional.jacobian(fk_single, qi)  # [2,3]
        J_list.append(Ji)
    J = torch.stack(J_list, dim=0)  # [B,2,3]
    return J.view(*orig_shape[:-1], 2, 3)


def compute_manipulability_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Compute manipulability for a batch of planar 3R joint configurations.

    w = sqrt(det(J J^T)), where J is [2,3], so JJ^T is [2,2].

    Args:
        q: [B,3] tensor of joint angles

    Returns:
        [B] tensor of manipulability scalars
    """
    J   = jacobian_torch(q)                          # [B,2,3]
    JJt = J @ J.transpose(-2, -1)                     # [B,2,2]
    det = torch.linalg.det(JJt).clamp(min=0.0)        # [B]
    return torch.sqrt(det)


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model + optimizer state to `path/ckpt_epoch.pt`.
    """
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }, os.path.join(path, f"ckpt_{epoch}.pt"))


def load_checkpoint(model, optimizer, ckpt_path):
    """
    Load model & optimizer state from a checkpoint file.
    Returns the epoch to resume from.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optim_state'])
    return ckpt['epoch']


class EarlyStopping:
    """
    Stop training when validation loss has not improved for `patience` epochs.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0

    def step(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
