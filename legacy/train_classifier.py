import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from config_classifier import DATA_PATH, OUTPUT_DIR, BATCH_SIZE, LR, NUM_EPOCHS, SEED
from dataset_classifier import get_loaders
from model_classifier import ReachabilityNet

def train_classifier():
    # reproducibility
    torch.manual_seed(SEED)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # data loaders
    train_loader, val_loader, test_loader = get_loaders(BATCH_SIZE)

    # model, optimizer, loss
    model = ReachabilityNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "runs"))

    best_val_acc = 0.0

    # history lists for plotting
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        train_preds, train_trues = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_trues.extend(yb.cpu().tolist())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc  = accuracy_score(train_trues, train_preds)

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
        val_acc  = accuracy_score(val_trues, val_preds)

        # record history
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Logging
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)

        # Checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "best_model.pt")
            )

    writer.close()

    # Plot training curves
    epochs = list(range(1, NUM_EPOCHS + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss_hist, label="Train Loss")
    plt.plot(epochs, val_loss_hist,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs. Epoch")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_acc_hist, label="Train Acc")
    plt.plot(epochs, val_acc_hist,   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs. Epoch")
    plt.show()

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().tolist()
            test_preds.extend(preds)
            test_trues.extend(yb.tolist())

    test_acc = accuracy_score(test_trues, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_classifier()
