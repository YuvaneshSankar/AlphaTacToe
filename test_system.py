"""
Quick test of the AlphaTacToe system
"""



import torch
import numpy as np
from game import TicTacToeGame
from mcts import MCTS
from neural_network_components.alphaTacToeNET import AlphaTacToeNet

def test_game():
    """Test the game implementation"""
    print("Testing game implementation...")
    game = TicTacToeGame()
    board = game.get_init_board()
    
    print("Initial board:")
    game.display_board(board)
    
    # Make a few moves
    board, _ = game.get_next_state(board, 1, 4)  # Center
    print("After move to center:")
    game.display_board(board)
    
    board, _ = game.get_next_state(board, -1, 0)  # Top-left
    print("After opponent move:")
    game.display_board(board)
    
    print("✅ Game test passed!")

def test_neural_network():
    """Test neural network prediction"""
    print("\\nTesting neural network...")
    net = AlphaTacToeNet()
    game = TicTacToeGame()
    board = game.get_init_board()
    
    # Test prediction
    action_probs, value = net.predict(board)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Action probabilities: {action_probs}")
    print(f"Value: {value}")
    print("✅ Neural network test passed!")

def test_mcts():
    """Test MCTS implementation"""
    print("\\nTesting MCTS...")
    game = TicTacToeGame()
    net = AlphaTacToeNet()
    args = {'num_simulations': 10}
    
    mcts = MCTS(game, net, args)
    board = game.get_init_board()
    
    action_probs = mcts.run(board, 1, temperature=1.0)
    print(f"MCTS action probabilities: {action_probs}")
    print(f"Sum of probabilities: {np.sum(action_probs)}")
    print("✅ MCTS test passed!")

def test_mini_training():
    """Test a mini training episode"""
    print("\\nTesting training episode...")
    
    from train import AlphaTacToeTrainer
    
    # Create trainer with minimal config
    config = {
        'num_iterations': 1,
        'num_episodes': 3,
        'num_mcts_sims': 5,
        'batch_size': 8,
        'epochs': 1,
        'learning_rate': 0.01,
        'eval_games': 2,
        'eval_threshold': 0.5,
    }
    
    trainer = AlphaTacToeTrainer(config)
    
    # Test self-play episode
    print("Testing self-play episode...")
    examples = trainer.execute_episode()
    print(f"Generated {len(examples)} training examples")
    
    # Test training if we have enough examples
    if len(examples) >= 3:
        print("Testing neural network training...")
        trainer.training_examples.extend(examples)
        trainer.train_network(examples[:3])
    
    print("✅ Mini training test passed!")

def main():
    """Run all tests"""
    print("🧪 Running AlphaTacToe System Tests")
    print("=" * 50)
    
    try:
        test_game()
        test_neural_network()
        test_mcts()
        test_mini_training()
        
        print("\\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The AlphaTacToe system is ready for training!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()