# AlphaTacToe 🎮

A complete implementation of the AlphaGo Zero algorithm applied to Tic-Tac-Toe, featuring a neural network with residual connections, Monte Carlo Tree Search (MCTS), and self-play training.

## 🏗️ Architecture

This implementation follows the AlphaGo Zero paper architecture adapted for Tic-Tac-Toe:

### Neural Network
- **Input**: 5-channel 3×3 representation of the board
  - Channel 0: Current player's pieces
  - Channel 1: Opponent's pieces  
  - Channel 2: Empty squares
  - Channel 3: Current player indicator
  - Channel 4: History plane
- **Architecture**: 
  - First convolutional layer (32 filters, 3×3)
  - Residual tower (10 residual blocks)
  - Policy head (outputs action probabilities)
  - Value head (outputs position evaluation)

### Monte Carlo Tree Search (MCTS)
- UCB1 exploration with neural network priors
- Temperature-based action selection
- Proper handling of player perspectives

### Training Loop
- Self-play data generation
- Neural network training on generated data
- Model evaluation and selection
- Iterative improvement

## 📁 Project Structure

```
alphaTacToe/
├── game.py                     # Tic-tac-toe game logic
├── mcts.py                     # Monte Carlo Tree Search implementation
├── train.py                    # Training loop
├── play.py                     # Command-line interface for playing
├── neural_network_components/  # Neural network components
│   ├── alphaTacToeNET.py      # Main network class
│   ├── first_covl_layer.py    # First convolution layer
│   ├── residual_tower.py      # Residual tower
│   ├── residual_block.py      # Individual residual blocks
│   ├── policy_head.py         # Policy head
│   └── value_head.py          # Value head
├── test_system.py             # System tests
├── test_cli.py                # CLI tests
├── quick_train.py             # Quick training demo
├── create_demo_model.py       # Create demo model
└── checkpoints/               # Saved models directory
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy
```

### 2. Test the System
```bash
# Run comprehensive system tests
PYTHONPATH=/home/{user}/alphaTacToe python test_system.py

# Test the neural network architecture
PYTHONPATH=/home/{user}/alphaTacToe python network_test.py
```

### 3. Train the Model
```bash
# Quick training (3 iterations for testing)
PYTHONPATH=/home/{user}/alphaTacToe python quick_train.py

# Full training (50 iterations)
PYTHONPATH=/home/{user}/alphaTacToe python train.py
```

### 4. Play Against the AI
```bash
# Create a demo model for testing
PYTHONPATH=/home/{user}/alphaTacToe python create_demo_model.py

# Launch the CLI game
PYTHONPATH=/home/{user}/alphaTacToe python play.py
```

## 🎮 How to Play

The CLI offers several game modes:

1. **Play against AI** - Challenge the trained model
2. **AI vs AI** - Watch two AIs battle it out
3. **Human vs Human** - Classic two-player mode
4. **Load different models** - Try different training checkpoints
5. **Adjust settings** - Change AI difficulty (MCTS simulations)

### Board Notation
Positions are numbered 0-8:
```
  0 1 2
0 0 1 2
1 3 4 5  
2 6 7 8
```

## ⚙️ Configuration

### Training Parameters
Edit the config in `train.py`:
- `num_iterations`: Number of training iterations
- `num_episodes`: Self-play games per iteration
- `num_mcts_sims`: MCTS simulations per move
- `batch_size`: Training batch size
- `learning_rate`: Neural network learning rate
- `eval_threshold`: Win rate needed to accept new model

### AI Difficulty
In the CLI, you can adjust AI strength:
- **Easy**: 10 MCTS simulations
- **Medium**: 25 MCTS simulations  
- **Hard**: 50 MCTS simulations
- **Expert**: 100 MCTS simulations
- **Custom**: Any number you choose

## 🧪 Testing

Run tests to verify everything works:

```bash
# Full system test
PYTHONPATH=/home/{user}/alphaTacToe python test_system.py

# CLI components test  
PYTHONPATH=/home/{user}/alphaTacToe python test_cli.py

# Neural network test
PYTHONPATH=/home/{user}/alphaTacToe python network_test.py
```

## 🔬 Technical Details

### Neural Network Features
- **Residual connections** for better gradient flow
- **Batch normalization** for training stability
- **Separate policy and value heads** following AlphaGo architecture
- **Softmax policy output** for action probabilities
- **Tanh value output** for position evaluation

### MCTS Features  
- **UCB1 exploration** with neural network priors
- **Temperature control** for exploration vs exploitation
- **Proper perspective handling** for two-player games
- **Efficient tree traversal** and backpropagation

### Training Features
- **Self-play data generation** with temperature annealing
- **Experience replay** with circular buffer
- **Model evaluation** against previous versions
- **Checkpoint saving** for model persistence
- **GPU acceleration** when available

## 📊 Expected Performance

With proper training (50+ iterations), the AI should:
- Beat random players consistently
- Learn basic tic-tac-toe strategy
- Avoid obvious mistakes
- Play optimally in many positions

Note: Tic-tac-toe is a solved game (draw with perfect play), so a perfectly trained AI should never lose and achieve draws against optimal opponents.

## 🚀 Next Steps

1. **Train longer**: Run more iterations for better performance
2. **Experiment with architecture**: Try different network sizes
3. **Tune hyperparameters**: Optimize learning rates and MCTS parameters
4. **Scale to larger games**: Apply to 4×4 tic-tac-toe or other games
5. **Add analysis tools**: Visualize training progress and game analysis

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Optimize performance
- Improve documentation

## 📜 License

This project is for educational purposes. The AlphaGo Zero algorithm is described in the DeepMind paper "Mastering the Game of Go without Human Knowledge".

---

**Happy Gaming! 🎮**

*Built with PyTorch and lots of coffee ☕*