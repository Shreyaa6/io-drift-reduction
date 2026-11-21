"""Compact sequence-model alternatives; all consume past-only IMU windows."""
import torch
from torch import nn
from .model import VelocityCNN


class ResidualBlock(nn.Module):
    def __init__(self, channels=48, dilation=1, dropout=.15):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2*dilation, dilation=dilation),
            nn.GroupNorm(6, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=2*dilation, dilation=dilation),
            nn.GroupNorm(6, channels))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class ResidualVelocity(nn.Module):
    def __init__(self, dilated=False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(7,48,7,padding=3), nn.GELU(),
            *[ResidualBlock(dilation=d) for d in ([1,2,4] if dilated else [1,1,1])])
        self.head = nn.Sequential(nn.Linear(96,64), nn.GELU(), nn.Dropout(.15), nn.Linear(64,3))

    def forward(self, x):
        h = self.encoder(x.transpose(1,2))
        return self.head(torch.cat([h.mean(-1),h[:,:,-1]],dim=-1))


class RecurrentVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(7,32,5,stride=2,padding=2),nn.GELU())
        self.gru = nn.GRU(32,64,num_layers=2,batch_first=True,dropout=.15)
        self.attention = nn.Linear(64,1)
        self.head = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,3))

    def forward(self, x):
        h,_ = self.gru(self.encoder(x.transpose(1,2)).transpose(1,2))
        pooled = (h*self.attention(h).softmax(dim=1)).sum(1)
        return self.head(torch.cat([pooled,h[:,-1]],dim=-1))


def make_model(architecture='cnn'):
    if architecture == 'cnn':
        return VelocityCNN()
    if architecture == 'resnet':
        return ResidualVelocity()
    if architecture == 'tcn':
        return ResidualVelocity(dilated=True)
    if architecture == 'gru':
        return RecurrentVelocity()
    raise ValueError(f'Unknown architecture: {architecture}')
