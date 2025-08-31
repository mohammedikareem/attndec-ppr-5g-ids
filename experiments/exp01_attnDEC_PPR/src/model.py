import torch, torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, in_dim=16, latent_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 2)
        )
    def forward(self, x):
        return self.net(x)
