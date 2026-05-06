import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from traj_seq_model import TrajectorySeqModel, save_traj_seq_model


def weighted_huber_loss(pred, target):
    # weights by horizon: +0.5s, +1.0s, +1.5s, +2.0s
    w = torch.tensor([1.0, 1.0, 1.3, 1.3, 1.7, 1.7, 2.0, 2.0], device=pred.device).view(1, 8)
    diff = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (diff * w).mean()


def ade_norm(pred, target):
    p = pred.view(-1, 4, 2)
    t = target.view(-1, 4, 2)
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1))
    return d.mean().item()


def main():
    data = np.load("processed_data.npz")
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_traj_train"], dtype=torch.float32)
    X_dev = torch.tensor(data["X_dev"], dtype=torch.float32)
    y_dev = torch.tensor(data["y_traj_dev"], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True, drop_last=True)
    dev_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectorySeqModel(input_size=X_train.shape[2], hidden_size=128, num_layers=2, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    best = float("inf")
    patience = 0
    max_patience = 12

    for epoch in range(60):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = weighted_huber_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        dev_losses = []
        dev_ades = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                dev_losses.append(weighted_huber_loss(pred, yb).item())
                dev_ades.append(ade_norm(pred, yb))
        dev_loss = float(np.mean(dev_losses))
        dev_ade = float(np.mean(dev_ades))
        sched.step(dev_loss)
        print(f"epoch {epoch+1:02d} dev_loss {dev_loss:.5f} dev_ade_norm {dev_ade:.5f}")

        if dev_loss < best:
            best = dev_loss
            patience = 0
            save_traj_seq_model(model, "traj_seq.pth")
            print("saved traj_seq.pth")
        else:
            patience += 1
            if patience >= max_patience:
                print("early stop")
                break


if __name__ == "__main__":
    main()

