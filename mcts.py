import torch
import math
import numpy as np

c_puct = 1.0

def ucb_score(parent, child):
    """Calculate UCB score for node selection"""
    exploration = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    if child.visit_count > 0:
        exploitation = child.value()
    else:
        exploitation = 0
    return exploitation + exploration

class Node:
    def __init__(self, prior, to_play):
        self.prior = prior # prior probability of selecting this state from its parent
        self.to_play = to_play # player to play in this state
        self.children = {} # lookup for all legal child positions
        self.visit_count = 0 # number of times this node has been visited
        self.value_sum = 0 # cumulative value of this node from all visits
        self.state = None # state of the game at this node

    def value(self):
        """Average value from perspective of player to play"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def expand(self, state, to_play, action_probs):
        """Expand the node by creating children for all legal actions"""
        self.state = state
        self.to_play = to_play
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = Node(prior=prob, to_play=-to_play)

    def select_child(self):
        """Select the child with highest UCB score"""
        best_score = float('-inf')
        best_action = -1
        best_child = None
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
    def expanded(self):
        """Check if node has been expanded"""
        return len(self.children) > 0

    def select_action(self, temperature):
        """Select action based on visit counts and temperature"""
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = np.array(list(self.children.keys()))
        
        if temperature == 0:
            # Select action with highest visit count
            max_visits = np.max(visit_counts)
            best_actions = actions[visit_counts == max_visits]
            return np.random.choice(best_actions)
        elif temperature == float('inf'):
            # Random selection
            return np.random.choice(actions)
        else:
            # Temperature-based selection
            visit_counts_temp = visit_counts ** (1 / temperature)
            probs = visit_counts_temp / visit_counts_temp.sum()
            return np.random.choice(actions, p=probs)

    def get_action_probs(self, temperature=1.0):
        """Get action probabilities based on visit counts"""
        visit_counts = np.zeros(9)  # 9 actions for tic-tac-toe
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count
        
        if temperature == 0:
            probs = np.zeros(9)
            best_action = np.argmax(visit_counts)
            probs[best_action] = 1.0
            return probs
        else:
            visit_counts_temp = visit_counts ** (1 / temperature)
            total = visit_counts_temp.sum()
            if total > 0:
                return visit_counts_temp / total
            else:
                # If no visits, return uniform over valid moves
                valid_actions = list(self.children.keys())
                probs = np.zeros(9)
                for action in valid_actions:
                    probs[action] = 1.0 / len(valid_actions)
                return probs


class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, state, to_play, temperature=1.0):
        """Run MCTS simulations and return action probabilities"""
        root = Node(0, to_play)

        # EXPAND ROOT NODE
        # Get canonical board from current player's perspective
        canonical_state = self.game.get_canonical_board(state, to_play)
        actions, value = self.model.predict(canonical_state)
        valid_moves = self.game.get_valid_moves(state)
        
        # Mask illegal moves
        action_probs = actions * valid_moves
        # Normalize
        total = np.sum(action_probs)
        if total > 0:
            action_probs = action_probs / total
        else:
            # If no valid moves, uniform over valid moves
            action_probs = valid_moves / np.sum(valid_moves) if np.sum(valid_moves) > 0 else valid_moves
        
        root.expand(state, to_play, action_probs)

        # Run simulations
        for _ in range(self.args.get('num_simulations', 100)):
            node = root
            search_path = [node]
            current_state = state
            current_player = to_play


            # SELECT: Traverse tree until we reach a leaf
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
                # Get next state
                current_state, current_player = self.game.get_next_state(current_state, current_player, action)

            # Check if game is terminal
            value = self.game.get_reward_for_player(current_state, to_play)
            
            if value is None:
                # EXPAND: Game not over, expand the leaf node
                canonical_state = self.game.get_canonical_board(current_state, current_player)
                actions, value = self.model.predict(canonical_state)
                valid_moves = self.game.get_valid_moves(current_state)
                
                action_probs = actions * valid_moves
                total = np.sum(action_probs)
                if total > 0:
                    action_probs = action_probs / total
                else:
                    action_probs = valid_moves / np.sum(valid_moves) if np.sum(valid_moves) > 0 else valid_moves
                
                node.expand(current_state, current_player, action_probs)
                
                # Value is from current player's perspective, convert to root player's perspective
                if current_player != to_play:
                    value = -value

            # BACKPROPAGATE: Update all nodes in search path
            self.backpropagate(search_path, value, to_play)

        return root.get_action_probs(temperature)

    def backpropagate(self, search_path, value, root_player):
        """Backpropagate value through the search path"""
        for node in reversed(search_path):
            # Value is always from root player's perspective
            # If current node's player is the root player, add value
            # If current node's player is opponent, subtract value
            node_value = value if node.to_play == root_player else -value
            node.value_sum += node_value
            node.visit_count += 1
