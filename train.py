"""
AlphaTacToe Training Loop

This implements the AlphaGo Zero training algorithm for Tic-Tac-Toe:
1. Self-play: Generate training data by playing games using MCTS
2. Training: Train the neural network on the generated data
3. Evaluation: Test the new model against the previous one
4. Iteration: Repeat the process
"""

import os
import time
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
from game import TicTacToeGame
from mcts import MCTS
from neural_network_components.alphaTacToeNET import AlphaTacToeNet


class AlphaTacToeDataset(Dataset):
    """Dataset for training AlphaTacToe network"""
    
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        board, pi, v = self.examples[idx]
        # Convert board to tensor representation
        board_tensor = AlphaTacToeNet().board_to_tensor(board)
        return board_tensor, torch.tensor(pi, dtype=torch.float32), torch.tensor(v, dtype=torch.float32)


class AlphaTacToeTrainer:
    """Main training class for AlphaTacToe"""
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'num_iterations': 100,          # Number of training iterations
            'num_episodes': 100,            # Number of self-play games per iteration
            'num_mcts_sims': 50,           # Number of MCTS simulations per move
            'temp_threshold': 15,           # Temperature threshold for action selection
            'checkpoint_dir': 'checkpoints',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,                   # Training epochs per iteration
            'max_examples': 200000,         # Maximum number of training examples to keep
            'eval_games': 40,              # Number of games for model evaluation
            'eval_threshold': 0.6,         # Win rate threshold for accepting new model
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self.game = TicTacToeGame()
        self.net = AlphaTacToeNet()
        self.net.to(self.config['device'])
        
        # Training data storage
        self.training_examples = deque(maxlen=self.config['max_examples'])
        
        # Create checkpoint directory
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        print(f"AlphaTacToe Trainer initialized on device: {self.config['device']}")
        print(f"Config: {self.config}")
    
    def execute_episode(self):
        """
        Execute one episode of self-play and return training examples.
        Returns list of (board, pi, v) tuples where:
        - board: game state
        - pi: MCTS action probabilities 
        - v: game outcome from current player's perspective
        """
        training_examples = []
        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0
        
        # Initialize MCTS
        mcts = MCTS(self.game, self.net, self.config)
        
        while True:
            episode_step += 1
            
            # Determine temperature for action selection
            temp = 1 if episode_step < self.config['temp_threshold'] else 0
            
            # Get action probabilities from MCTS
            pi = mcts.run(board, current_player, temperature=temp)
            
            # Store training example (outcome will be filled later)
            training_examples.append([board.copy(), current_player, pi, None])
            
            # Select action based on probabilities
            if temp == 0:
                action = np.argmax(pi)
            else:
                action = np.random.choice(len(pi), p=pi)
            
            # Make move
            board, current_player = self.game.get_next_state(board, current_player, action)
            
            # Check if game is over
            reward = self.game.get_reward_for_player(board, current_player)
            
            if reward is not None:
                # Game ended, assign rewards to all positions
                final_examples = []
                for hist_board, hist_player, hist_pi, _ in training_examples:
                    # Reward is from the perspective of hist_player
                    if reward == 0:  # Draw
                        outcome = 0
                    elif hist_player == current_player:
                        outcome = -reward  # Current player lost/won, so hist_player won/lost
                    else:
                        outcome = reward   # Current player lost/won, so hist_player won/lost
                    
                    final_examples.append((hist_board, hist_pi, outcome))
                
                return final_examples
    
    def train_network(self, examples):
        """Train the neural network on the provided examples"""
        print(f"Training network on {len(examples)} examples...")
        
        # Create dataset and dataloader
        dataset = AlphaTacToeDataset(examples)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        # Setup optimizer and loss functions
        optimizer = optim.Adam(self.net.parameters(), lr=self.config['learning_rate'])
        policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        value_loss_fn = nn.MSELoss()
        
        self.net.train()
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(self.config['epochs']):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            
            for batch_idx, (boards, target_pis, target_vs) in enumerate(dataloader):
                boards = boards.to(self.config['device'])
                target_pis = target_pis.to(self.config['device'])
                target_vs = target_vs.to(self.config['device'])
                
                # Forward pass
                pred_pis, pred_vs = self.net(boards)
                pred_vs = pred_vs.squeeze()
                
                # Calculate losses
                policy_loss = policy_loss_fn(torch.log(pred_pis + 1e-8), target_pis)
                value_loss = value_loss_fn(pred_vs, target_vs)
                total_loss = policy_loss + value_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
            
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Policy Loss: {epoch_policy_loss/len(dataloader):.4f}, "
                  f"Value Loss: {epoch_value_loss/len(dataloader):.4f}")
        
        avg_policy_loss = total_policy_loss / (self.config['epochs'] * len(dataloader))
        avg_value_loss = total_value_loss / (self.config['epochs'] * len(dataloader))
        
        print(f"Training completed. Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")
        
        return avg_policy_loss, avg_value_loss
    
    def evaluate_model(self, new_net):
        """Evaluate new model against current model"""
        print("Evaluating new model...")
        
        wins = 0
        draws = 0
        losses = 0
        
        for game_num in range(self.config['eval_games']):
            # Alternate who goes first
            if game_num % 2 == 0:
                players = [self.net, new_net]  # Current model goes first
            else:
                players = [new_net, self.net]  # New model goes first
            
            board = self.game.get_init_board()
            current_player = 1
            
            while True:
                # Get action from appropriate model
                player_idx = 0 if current_player == 1 else 1
                model = players[player_idx]
                
                # Use MCTS with the model to get action
                mcts = MCTS(self.game, model, {'num_simulations': self.config['num_mcts_sims']})
                pi = mcts.run(board, current_player, temperature=0)
                action = np.argmax(pi)
                
                # Make move
                board, current_player = self.game.get_next_state(board, current_player, action)
                
                # Check if game is over
                reward = self.game.get_reward_for_player(board, 1)  # From player 1's perspective
                
                if reward is not None:
                    if reward == 0:  # Draw
                        draws += 1
                    elif (reward == 1 and players[0] == new_net) or (reward == -1 and players[1] == new_net):
                        wins += 1  # New model won
                    else:
                        losses += 1  # Current model won
                    break
        
        win_rate = wins / self.config['eval_games']
        print(f"Evaluation results: Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"New model win rate: {win_rate:.3f}")
        
        return win_rate
    
    def save_checkpoint(self, iteration, examples):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.net.state_dict(),
            'examples': list(examples),
            'config': self.config
        }
        
        filepath = os.path.join(self.config['checkpoint_dir'], f'checkpoint_{iteration}.pth')
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config['device'])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['iteration'], checkpoint['examples']
    
    def train(self):
        """Main training loop"""
        print("Starting AlphaTacToe training...")
        
        for iteration in range(self.config['num_iterations']):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration + 1}/{self.config['num_iterations']}")
            print(f"{'='*50}")
            
            # Self-play phase
            print("Phase 1: Self-play data generation")
            iteration_examples = []
            
            for episode in range(self.config['num_episodes']):
                if episode % 10 == 0:
                    print(f"Episode {episode + 1}/{self.config['num_episodes']}")
                
                episode_examples = self.execute_episode()
                iteration_examples.extend(episode_examples)
            
            # Add new examples to training pool
            self.training_examples.extend(iteration_examples)
            print(f"Generated {len(iteration_examples)} new training examples")
            print(f"Total training examples: {len(self.training_examples)}")
            
            # Training phase
            print("\nPhase 2: Neural network training")
            if len(self.training_examples) > self.config['batch_size']:
                # Sample training examples
                train_examples = random.sample(list(self.training_examples), 
                                             min(len(self.training_examples), 10000))
                
                # Create new network for training
                new_net = AlphaTacToeNet()
                new_net.to(self.config['device'])
                new_net.load_state_dict(self.net.state_dict())  # Copy current weights
                
                # Train the network
                policy_loss, value_loss = self.train_network(train_examples)
                
                # Evaluation phase
                print("\nPhase 3: Model evaluation")
                if iteration > 0:  # Skip evaluation for first iteration
                    win_rate = self.evaluate_model(new_net)
                    
                    if win_rate > self.config['eval_threshold']:
                        print(f"New model accepted! (win rate: {win_rate:.3f})")
                        self.net = new_net
                    else:
                        print(f"New model rejected (win rate: {win_rate:.3f} < {self.config['eval_threshold']})")
                else:
                    print("Skipping evaluation for first iteration")
                    self.net = new_net
            
            # Save checkpoint
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(iteration + 1, self.training_examples)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print("="*50)
        
        # Save final model
        final_path = os.path.join(self.config['checkpoint_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'config': self.config
        }, final_path)
        print(f"Final model saved: {final_path}")


def main():
    """Main training function"""
    
    # ================================================================================================
    # OPTIMAL TRAINING PARAMETERS FOR BEST RL PERFORMANCE
    # ================================================================================================
    # 
    # FOR PRODUCTION-LEVEL TRAINING (Best Performance):
    # - num_iterations: 200-500     (More iterations = better convergence)
    # - num_episodes: 100-200       (More self-play games = more diverse training data)
    # - num_mcts_sims: 100-200      (More simulations = stronger play during training)
    # - batch_size: 128-256         (Larger batches = more stable gradients)
    # - epochs: 10-20               (More epochs = better learning from each batch)
    # - learning_rate: 0.0005-0.001 (Lower LR = more stable training)
    # - eval_games: 50-100          (More evaluation games = better model selection)
    # - eval_threshold: 0.55-0.6    (Higher threshold = only accept significantly better models)
    # 
    # ESTIMATED TRAINING TIME:
    # - GPU (RTX 3080/4080): ~6-12 hours for 200 iterations
    # - GPU (RTX 3060/4060): ~8-16 hours for 200 iterations  
    # - CPU only: ~24-48 hours for 200 iterations (dont try brother :) )
    # 
    # FOR QUICK TESTING/DEVELOPMENT:
    # - Use the config below for faster iteration cycles
    # - Good for debugging and initial validation
    # 
    # FOR RESEARCH/EXPERIMENTATION:
    # - num_iterations: 1000+       (For publication-quality results)
    # - num_mcts_sims: 400-800      (AlphaGo Zero used 800 simulations)
    # - Training time: 2-7 days on high-end GPU
    # ================================================================================================
    
    # Current configuration (balanced for development/testing)
    config = {
        'num_iterations': 50,         
        'num_episodes': 50,            
        'num_mcts_sims': 25,         
        'batch_size': 64,            
        'epochs': 5,                  
        'learning_rate': 0.001,      
        'eval_games': 20,            
        'eval_threshold': 0.55,      
    }

    #I wrote these for me to remember later and make it easier instead of scrolling up and down
    #I hope this helps anyone who reads this code in the future (if anyone does :) )
    
    #PRODUCTION TRAINING (strong AI, ~8-16 hours on good GPU):
    # config = {
    #     'num_iterations': 300,
    #     'num_episodes': 150, 
    #     'num_mcts_sims': 150,
    #     'batch_size': 128,
    #     'epochs': 15,
    #     'learning_rate': 0.0007,
    #     'eval_games': 60,
    #     'eval_threshold': 0.58,
    # }
    
    # RESEARCH TRAINING (publication-quality, ~2-4 days on high-end GPU):
    # config = {
    #     'num_iterations': 1000,
    #     'num_episodes': 200,
    #     'num_mcts_sims': 400,
    #     'batch_size': 256,
    #     'epochs': 20,
    #     'learning_rate': 0.0005,
    #     'eval_games': 100,
    #     'eval_threshold': 0.6,
    # }
    
    # Create trainer and start training
    trainer = AlphaTacToeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()