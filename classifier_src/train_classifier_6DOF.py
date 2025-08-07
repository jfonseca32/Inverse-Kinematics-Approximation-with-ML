"""
6DOF_train_classifier.py

Train a 6-DOF reachability/conditioning classifier on 7D pose inputs:
  [x, y, z, qx, qy, qz, qw] → labels {0,1,2,3}.
Includes a ReduceLROnPlateau scheduler to drop the learning rate when
validation accuracy plateaus.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from config_classifier_6DOF import (
    DATA_PATH,
    OUTPUT_DIR,
    BATCH_SIZE,
    LR,
    NUM_EPOCHS,
    SEED
)
from dataset_classifier_6DOF import get_loaders
from model_classifier_6DOF import ReachabilityNet6D as ReachabilityNet


def train_classifier():
    # Reproducibility & device 
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output directory & TensorBoard
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "runs"))

    # Data loaders 
    train_loader, val_loader, test_loader = get_loaders(BATCH_SIZE)

    # Model, optimizer, loss
    model     = ReachabilityNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",        # monitor validation accuracy
        factor=0.5,        # reduce LR by a factor of 0.5
        patience=5        # wait 5 epochs with no improvement
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        epoch_losses = []
        all_preds, all_trues = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                 
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_trues.extend(yb.cpu().tolist())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc  = accuracy_score(all_trues, all_preds)

        # Validate
        model.eval()
        val_losses = []
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss   = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_trues.extend(yb.cpu().tolist())

        val_loss = sum(val_losses) / len(val_losses)
        val_acc  = accuracy_score(val_val_true := val_trues, val_preds)

        # Scheduler step 
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"Reducing learning rate: {old_lr:.2e} → {new_lr:.2e}")

        # Logging & checkpoint
        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("LR",         optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "best_model_6DOF.pt")
            )

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

    writer.close()

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model_6DOF.pt")))
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().tolist()
            test_preds.extend(preds)
            test_trues.extend(yb.tolist())

    test_acc = accuracy_score(test_trues, test_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # # Plot training curves 
    # epochs = list(range(1, NUM_EPOCHS + 1))
    # # first figure (loss)
    # plt.figure(figsize=(8, 4))
    # plt.plot(epochs, train_loss_hist, label="Train Loss")
    # plt.plot(epochs, val_loss_hist,   label="Val Loss")
    # plt.xlabel("Epoch"); plt.ylabel("Loss")
    # plt.legend(); plt.title("Loss vs Epoch")

    # # second figure (accuracy)
    # plt.figure(figsize=(8, 4))
    # plt.plot(epochs, train_acc_hist, label="Train Acc")
    # plt.plot(epochs, val_acc_hist,   label="Val Acc")
    # plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    # plt.legend(); plt.title("Accuracy vs Epoch")

    # plt.show()

    # confusion matrix plot
    labels = [0, 1, 2, 3]
    display_labels = [
        "Unreach", 
        "Near-sing", 
        "Well-cond", 
        "High-cond"
    ]

    cm = confusion_matrix(test_trues, test_preds, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )
    disp.plot(
        ax=ax,
        cmap=plt.cm.Blues, # type: ignore
        colorbar=False
    )
    ax.set_title("6DOF Reachability Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_classifier()