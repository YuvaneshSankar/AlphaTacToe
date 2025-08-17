import torch
import torch.nn as nn

# Import your component classes
from neural_network_components.first_cov_layer import FirstConvLayer
from neural_network_components.residual_block import ResidualBlock
from neural_network_components.residual_tower import ResidualTower
from neural_network_components.policy_head import PolicyHead
from neural_network_components.value_head import ValueHead


class AlphaTacToeNet(nn.Module):
    def __init__(self, in_channels=5, channels=32, n_blocks=10):
        super(AlphaTacToeNet, self).__init__()
        self.first_conv = FirstConvLayer(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.residual_tower = ResidualTower(channels, n_blocks=n_blocks)
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.residual_tower(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v


if __name__ == "__main__":
    net = AlphaTacToeNet()
    dummy = torch.randn(2, 5, 3, 3)   # batch=2, 5 planes, 3x3 board
    p, v = net(dummy)
    print("Policy shape:", p.shape)   # expect (2, 9)
    print("Value shape:", v.shape)    # expect (2, 1)
