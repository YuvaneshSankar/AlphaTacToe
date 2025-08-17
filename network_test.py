import torch
from neural_network_components.alphaTacToeNET import AlphaTacToeNet

def test_alphatacnet():
    # Initialize the network
    net = AlphaTacToeNet(in_channels=5, channels=32, n_blocks=3)

    # Create dummy batch: 2 positions, 5 planes, 3x3 board
    dummy_input = torch.randn(2, 5, 3, 3)

    # Forward pass
    policy, value = net(dummy_input)

    # Print outputs
    print("Input shape:", dummy_input.shape)
    print("Policy shape (expect [2, 9]):", policy.shape)
    print("Value shape (expect [2, 1]):", value.shape)

    # Sanity checks
    assert policy.shape == (2, 9), f"Policy shape mismatch: {policy.shape}"
    assert value.shape == (2, 1), f"Value shape mismatch: {value.shape}"
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_alphatacnet()
