import torch
import torch.nn as nn


class TrajectorySeqModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 8),
        )

    def forward(self, x):
        _, h = self.gru(x)
        z = h[-1]
        return self.head(z)


def save_traj_seq_model(model, path):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_size": model.input_size,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
            },
        },
        path,
    )


def load_traj_seq_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = TrajectorySeqModel(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model

