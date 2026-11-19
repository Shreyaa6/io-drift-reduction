import torch
from torch import nn


class VelocityCNN(nn.Module):
    """Temporal convolutions over past windows; no attitude/reference inputs."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(7,32,7,padding=3), nn.ReLU(),
            nn.Conv1d(32,64,5,stride=2,padding=2), nn.ReLU(),
            nn.Conv1d(64,64,5,stride=2,padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64,3))

    def forward(self, x):
        h = self.features(x.transpose(1,2))
        return self.head(torch.cat([h.mean(-1), h[:,:,-1]], dim=-1))
