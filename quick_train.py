"""
Quick training session for AlphaTacToe - just a few iterations to verify everything works
"""

from train import AlphaTacToeTrainer

def quick_train():
    """Run a quick training session"""
    # Very small config for quick testing
    config = {
        'num_iterations': 3,     # Just 3 iterations
        'num_episodes': 10,      # 10 games per iteration
        'num_mcts_sims': 15,     # 15 MCTS simulations
        'batch_size': 32,
        'epochs': 2,             # 2 training epochs
        'learning_rate': 0.01,
        'eval_games': 6,         # 6 evaluation games
        'eval_threshold': 0.5,
    }
    
    print("Starting quick training session...")
    trainer = AlphaTacToeTrainer(config)
    trainer.train()

if __name__ == "__main__":
    quick_train()