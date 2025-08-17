import torch.nn as nn
from neural_network_components.residual_block import ResidualBlock

class ResidualTower(nn.Module):
    def __init__(self, channels, n_blocks=5):
        super(ResidualTower, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)
