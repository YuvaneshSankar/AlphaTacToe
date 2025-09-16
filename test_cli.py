"""
Quick CLI test 
"""


from game import TicTacToeGame
from play import AlphaTacToePlayer

def test_cli_components():
    """Test CLI components"""
    print("Testing CLI components...")
    
    # Test AI player loading
    ai_player = AlphaTacToePlayer("checkpoints/demo_model.pth", num_mcts_sims=10)
    
    # Test game
    game = TicTacToeGame()
    board = game.get_init_board()
    
    print("Testing AI vs AI game...")
    current_player = 1
    move_count = 0
    
    while move_count < 9:  # Max 9 moves in tic-tac-toe
        print(f"\\nMove {move_count + 1}, Player {'X' if current_player == 1 else 'O'}")
        game.display_board(board)
        
        # Get AI move
        action = ai_player.get_action(board, current_player)
        print(f"AI chooses position {action}")
        
        # Make move
        board, current_player = game.get_next_state(board, current_player, action)
        move_count += 1
        
        # Check if game is over
        reward = game.get_reward_for_player(board, 1)
        if reward is not None:
            print("\\nFinal board:")
            game.display_board(board)
            if reward == 1:
                print("Player 1 (X) wins!")
            elif reward == -1:
                print("Player 2 (O) wins!")
            else:
                print("It's a draw!")
            break
    
    print("\\n✅ CLI components test passed!")
    print("\\nYou can now run: python play.py")
    print("To play against the AI interactively!")

if __name__ == "__main__":
    test_cli_components()