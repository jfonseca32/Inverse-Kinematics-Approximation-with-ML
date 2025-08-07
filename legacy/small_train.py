import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from config            import *
from dataset           import get_dataloaders, IKDataset
from model             import IKNet
from utils             import save_checkpoint, EarlyStopping, compute_manipulability_torch

def small_train(n_small: int = 500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the full loaders
    train_loader_full, val_loader, _ = get_dataloaders(test_size=0.1, val_size=0.1)

    # build a small training loader by taking the first n_small samples
    small_subset = Subset(train_loader_full.dataset, list(range(n_small)))
    small_loader = DataLoader(small_subset, batch_size=BATCH_SIZE, shuffle=True)

    # model
    model = IKNet().to(device)

    # optimizer & scheduler (stepped per batch of the small loader)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    # loss, early-stopper, logging
    criterion = nn.SmoothL1Loss()
    # stopper   = EarlyStopping(patience=EARLY_STOPPING)
    writer    = SummaryWriter()

    for epoch in range(1, 350 + 1):
        # Training on small subset
        model.train()
        train_loss_small = 0.0
        for x_batch, y_batch in small_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            

            train_loss_small += loss.item() * x_batch.size(0)

        train_loss_small /= len(small_loader.dataset)
        scheduler.step()

        # # ——— Validation on full val set ———
        # model.eval()
        # val_loss    = 0.0
        # manip_ratio = 0.0
        # n_val       = 0
        # with torch.no_grad():
        #     for x_val, y_val in val_loader:
        #         x_val, y_val = x_val.to(device), y_val.to(device)
        #         y_pred = model(x_val)

        #         # MSE on sin/cos vectors
        #         val_loss += criterion(y_pred, y_val).item() * x_val.size(0)

        #         # split into sin vs cos for each joint, then recover angles
        #         y_pred_np = y_pred.cpu().numpy()
        #         y_val_np  = y_val .cpu().numpy()
        #         sin_pred  = y_pred_np[:, 0::2]
        #         cos_pred  = y_pred_np[:, 1::2]
        #         sin_true  = y_val_np [:, 0::2]
        #         cos_true  = y_val_np [:, 1::2]

        #         theta_pred = np.arctan2(sin_pred, cos_pred).astype(np.float32)
        #         theta_true = np.arctan2(sin_true, cos_true).astype(np.float32)

        #         theta_pred = torch.from_numpy(theta_pred).to(device)
        #         theta_true = torch.from_numpy(theta_true).to(device)

        #         w_pred = compute_manipulability_torch(theta_pred)
        #         w_true = compute_manipulability_torch(theta_true)
        #         manip_ratio += (w_pred / w_true).sum().item()
        #         n_val       += x_val.size(0)

        # val_loss    /= n_val
        # manip_ratio /= n_val

        # logging
        print(
            f"Epoch {epoch:3d}  "
            f"Train Loss: {train_loss_small:.4f}  ",
            # f"Val Loss: {val_loss:.4f}  "
            # f"Manip Ratio: {manip_ratio:.4f}",
            flush=True
        )
        writer.add_scalar("Loss/train_small", train_loss_small, epoch)
        # writer.add_scalar("Loss/val",         val_loss,          epoch)
        # writer.add_scalar("Metric/manip_ratio", manip_ratio,     epoch)

        save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR)
        # if stopper.step(val_loss):
        #     print(f"Early stopping at epoch {epoch}")
        #     break

    writer.close()

if __name__ == "__main__":
    small_train(n_small=1000)
