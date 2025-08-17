import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, in_channels, mid_channels=1):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn   = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(mid_channels * 3 * 3, 50, bias=False) 
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(50, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

