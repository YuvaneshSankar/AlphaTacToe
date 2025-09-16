import torch
import torch.nn as nn
import numpy as np

# Import your component classes
from neural_network_components.first_covl_layer import FirstConvLayer
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

    def predict(self, board):
        """
        Prediction method for MCTS.
        board: numpy array of shape (3, 3)
        Returns: (action_probs, value) where action_probs is shape (9,) and value is scalar
        """
        self.eval()
        with torch.no_grad():
            # Convert board to tensor and add necessary dimensions
            if isinstance(board, np.ndarray):
                # Convert 3x3 board to 5-plane representation
                board_tensor = self.board_to_tensor(board)
            else:
                board_tensor = board
            
            # Add batch dimension if not present
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)
            
            # Move tensor to same device as model
            device = next(self.parameters()).device
            board_tensor = board_tensor.to(device)
            
            # Forward pass
            policy_logits, value = self.forward(board_tensor)
            
            # Convert to numpy and remove batch dimension
            action_probs = policy_logits.squeeze(0).cpu().numpy()
            value = value.squeeze().cpu().numpy().item()
            
            return action_probs, value

    def board_to_tensor(self, board):
        """
        Convert 3x3 numpy board to 5-channel tensor representation.
        board: (3, 3) numpy array with values -1, 0, 1
        Returns: (5, 3, 3) tensor
        """
        tensor = torch.zeros(5, 3, 3, dtype=torch.float32)
        
        # Plane 0: Current player's pieces (assuming current player is 1)
        tensor[0] = torch.tensor(board == 1, dtype=torch.float32)
        
        # Plane 1: Opponent's pieces
        tensor[1] = torch.tensor(board == -1, dtype=torch.float32)
        
        # Plane 2: Empty squares
        tensor[2] = torch.tensor(board == 0, dtype=torch.float32)
        
        # Plane 3: Current player indicator (all 1s for player 1)
        tensor[3] = torch.ones(3, 3, dtype=torch.float32)
        
        # Plane 4: History plane (simplified - just copy current state)
        tensor[4] = tensor[0] + tensor[1]
        
        return tensor


if __name__ == "__main__":
    net = AlphaTacToeNet()
    dummy = torch.randn(2, 5, 3, 3)   # batch=2, 5 planes, 3x3 board
    p, v = net(dummy)
    print("Policy shape:", p.shape)   # expect (2, 9)
    print("Value shape:", v.shape)    # expect (2, 1)
