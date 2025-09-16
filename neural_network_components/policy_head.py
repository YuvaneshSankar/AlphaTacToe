import torch.nn as nn



class PolicyHead(nn.Module):
    def __init__(self, in_channels, mid_channels=2):
        super(PolicyHead, self).__init__()
        self.conv=nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False) #2 filters
        self.bn=nn.BatchNorm2d(mid_channels)
        self.relu=nn.ReLU(inplace=True)
        self.flatten=nn.Flatten()
        self.fc=nn.Linear(mid_channels * 3 * 3, 9, bias=False)  # we have 9 actions in tic-tac-toe 
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    



    
