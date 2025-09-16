"""
Create a simple demo model for testing the CLI
"""

import torch
import os
from neural_network_components.alphaTacToeNET import AlphaTacToeNet

def create_demo_model():
    """Create a simple demo model"""
    print("Creating demo model...")
    
    # Create model
    net = AlphaTacToeNet()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save as demo model
    demo_path = 'checkpoints/demo_model.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'config': {
            'num_mcts_sims': 25,
            'device': 'cpu'
        }
    }, demo_path)
    
    print(f"Demo model saved to: {demo_path}")
    print("This is a randomly initialized model for testing purposes.")
    print("Run the training to get a properly trained model!")

if __name__ == "__main__":
    create_demo_model()