"""
AlphaTacToe Command Line Interface

Play Tic-Tac-Toe against the trained AlphaTacToe AI!
"""

import os
import torch
import numpy as np
from game import TicTacToeGame
from mcts import MCTS
from neural_network_components.alphaTacToeNET import AlphaTacToeNet


class AlphaTacToePlayer:
    """AI player using trained AlphaTacToe model"""
    
    def __init__(self, model_path, num_mcts_sims=50):
        self.game = TicTacToeGame()
        self.num_mcts_sims = num_mcts_sims
        
        # Load the trained model
        self.net = AlphaTacToeNet()
        self.load_model(model_path)
        self.net.eval()
        
        print(f"AlphaTacToe AI loaded with {num_mcts_sims} MCTS simulations")
    
    def load_model(self, model_path):
        """Load trained model weights"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.net.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.net.load_state_dict(checkpoint)
            print(f"Model loaded from: {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using randomly initialized network")
    
    def get_action(self, board, player):
        """Get AI's action using MCTS"""
        mcts = MCTS(self.game, self.net, {'num_simulations': self.num_mcts_sims})
        pi = mcts.run(board, player, temperature=0)
        return np.argmax(pi)


class TicTacToeCLI:
    """Command line interface for playing Tic-Tac-Toe"""
    
    def __init__(self):
        self.game = TicTacToeGame()
        self.ai_player = None
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*50)
        print("🎮 ALPHATACTOE - TIC TAC TOE AI 🎮")
        print("="*50)
        print("1. Play against AI")
        print("2. AI vs AI")
        print("3. Human vs Human")
        print("4. Load different AI model")
        print("5. Settings")
        print("6. Exit")
        print("="*50)
    
    def display_board_with_positions(self):
        """Display empty board with position numbers"""
        print("\nPosition numbers:")
        print("  0 1 2")
        for i in range(3):
            row_str = f"{i} "
            for j in range(3):
                pos = i * 3 + j
                row_str += f"{pos} "
            print(row_str)
        print()
    
    def get_human_move(self, board, player):
        """Get move from human player"""
        valid_moves = self.game.get_valid_moves(board)
        valid_positions = [i for i, valid in enumerate(valid_moves) if valid]
        
        while True:
            try:
                print(f"\nPlayer {'X' if player == 1 else 'O'}, enter your move (0-8): ", end="")
                move = int(input())
                
                if move in valid_positions:
                    return move
                else:
                    print(f"Invalid move! Valid positions: {valid_positions}")
            except ValueError:
                print("Please enter a number between 0 and 8")
            except KeyboardInterrupt:
                print("\nGame interrupted!")
                return None
    
    def play_game(self, player1_type, player2_type):
        """
        Play a single game
        player_type can be: 'human', 'ai'
        """
        board = self.game.get_init_board()
        current_player = 1
        move_count = 0
        
        print(f"\n🎮 Starting new game!")
        print(f"Player 1 (X): {player1_type}")
        print(f"Player 2 (O): {player2_type}")
        
        if player1_type == 'human' or player2_type == 'human':
            self.display_board_with_positions()
        
        while True:
            print(f"\nMove {move_count + 1}")
            self.game.display_board(board)
            
            # Determine current player type
            current_player_type = player1_type if current_player == 1 else player2_type
            
            # Get move based on player type
            if current_player_type == 'human':
                action = self.get_human_move(board, current_player)
                if action is None:  # Game interrupted
                    return None
            elif current_player_type == 'ai':
                if self.ai_player is None:
                    print("Error: AI player not loaded!")
                    return None
                print(f"AI (Player {'X' if current_player == 1 else 'O'}) is thinking...")
                action = self.ai_player.get_action(board, current_player)
                print(f"AI chooses position {action}")
            
            # Make move
            board, current_player = self.game.get_next_state(board, current_player, action)
            move_count += 1
            
            # Check if game is over
            reward = self.game.get_reward_for_player(board, 1)
            
            if reward is not None:
                print(f"\nFinal board:")
                self.game.display_board(board)
                
                if reward == 1:
                    winner = "Player 1 (X)"
                elif reward == -1:
                    winner = "Player 2 (O)"
                else:
                    winner = "It's a draw!"
                
                print(f"🎉 Game Over! {winner}")
                return winner
    
    def load_ai_model(self):
        """Load AI model"""
        print("\nAvailable model files:")
        checkpoint_dir = "checkpoints"
        
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if files:
                for i, file in enumerate(files):
                    print(f"{i+1}. {file}")
                
                try:
                    choice = int(input("Enter file number (or 0 for manual path): "))
                    if choice == 0:
                        model_path = input("Enter full path to model file: ")
                    elif 1 <= choice <= len(files):
                        model_path = os.path.join(checkpoint_dir, files[choice-1])
                    else:
                        print("Invalid choice!")
                        return
                except ValueError:
                    print("Invalid input!")
                    return
            else:
                model_path = input("No models found in checkpoints/. Enter full path: ")
        else:
            model_path = input("Checkpoints directory not found. Enter full path to model: ")
        
        # Load the model
        num_sims = self.get_ai_difficulty()
        self.ai_player = AlphaTacToePlayer(model_path, num_sims)
    
    def get_ai_difficulty(self):
        """Get AI difficulty (number of MCTS simulations)"""
        print("\\nChoose AI difficulty:")
        print("1. Easy (10 simulations)")
        print("2. Medium (25 simulations)")
        print("3. Hard (50 simulations)")
        print("4. Expert (100 simulations)")
        print("5. Custom")
        
        try:
            choice = int(input("Enter choice (1-5): "))
            if choice == 1:
                return 10
            elif choice == 2:
                return 25
            elif choice == 3:
                return 50
            elif choice == 4:
                return 100
            elif choice == 5:
                return int(input("Enter number of simulations: "))
            else:
                print("Invalid choice, using medium difficulty")
                return 25
        except ValueError:
            print("Invalid input, using medium difficulty")
            return 25
    
    def settings_menu(self):
        """Settings menu"""
        while True:
            print("\\n⚙️  SETTINGS")
            print("1. Change AI difficulty")
            print("2. Reload AI model")
            print("3. Back to main menu")
            
            try:
                choice = int(input("Enter choice: "))
                if choice == 1:
                    if self.ai_player:
                        self.ai_player.num_mcts_sims = self.get_ai_difficulty()
                        print(f"AI difficulty updated to {self.ai_player.num_mcts_sims} simulations")
                    else:
                        print("No AI model loaded!")
                elif choice == 2:
                    self.load_ai_model()
                elif choice == 3:
                    break
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Invalid input!")
    
    def run(self):
        """Main CLI loop"""
        print("Welcome to AlphaTacToe!")
        
        # Try to load default model
        default_models = [
            "checkpoints/final_model.pth",
            "checkpoints/checkpoint_50.pth",
            "final_model.pth"
        ]
        
        for model_path in default_models:
            if os.path.exists(model_path):
                self.ai_player = AlphaTacToePlayer(model_path)
                break
        
        if self.ai_player is None:
            print("No default model found. Please load a model manually.")
        
        while True:
            self.display_menu()
            
            try:
                choice = int(input("Enter your choice: "))
                
                if choice == 1:
                    # Play against AI
                    if self.ai_player is None:
                        print("Please load an AI model first!")
                        self.load_ai_model()
                        if self.ai_player is None:
                            continue
                    
                    print("Who goes first?")
                    print("1. You (X)")
                    print("2. AI (X)")
                    
                    try:
                        first = int(input("Enter choice: "))
                        if first == 1:
                            self.play_game('human', 'ai')
                        elif first == 2:
                            self.play_game('ai', 'human')
                        else:
                            print("Invalid choice!")
                    except ValueError:
                        print("Invalid input!")
                
                elif choice == 2:
                    # AI vs AI
                    if self.ai_player is None:
                        print("Please load an AI model first!")
                        self.load_ai_model()
                        if self.ai_player is None:
                            continue
                    
                    print("Watching AI vs AI game...")
                    self.play_game('ai', 'ai')
                
                elif choice == 3:
                    # Human vs Human
                    self.play_game('human', 'human')
                
                elif choice == 4:
                    # Load different model
                    self.load_ai_model()
                
                elif choice == 5:
                    # Settings
                    self.settings_menu()
                
                elif choice == 6:
                    # Exit
                    print("Thanks for playing AlphaTacToe! 🎮")
                    break
                
                else:
                    print("Invalid choice! Please enter 1-6.")
            
            except ValueError:
                print("Invalid input! Please enter a number.")
            except KeyboardInterrupt:
                print("\\nThanks for playing AlphaTacToe! 🎮")
                break


def main():
    """Main function"""
    cli = TicTacToeCLI()
    cli.run()


if __name__ == "__main__":
    main()