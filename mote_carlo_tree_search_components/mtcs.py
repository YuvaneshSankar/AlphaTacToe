import torch
import math
import numpy as np


def ucb_score(parent,child):
    exploration=c_puct*child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    if child.visit_count >0:
        exploitation=child.value()
    else:
        exploitation=0
    return exploitation + exploration

class Node:
    def __init__(self, prior, to_play):
        self.prior = prior # prioir probability of selecting this state from its parent
        self.to_play = to_play # player to play in this state
        self.children = {} # lookup for all legal child positions
        self.visit_count = 0 # number of times this node has been visited
        self.value_sum = 0 # cumulative value of this node from all visits
        self.state = None # state of the game at this node

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0  #this is the Q(s,a) in that puct formula 

    def expand(self,state,to_play,action_prob):
        self.state = state
        self.to_play = to_play
        for a,prob in enumerate(action_prob):
            if prob!=0:
                self.children[a]=Node(prior=prob,to_play=-to_play)  # create a new child node for each legal action with its prior probability
    
    def select_child(self):
        best_score= float('-inf')
        best_action = -1
        best_child = None
        for action,child in self.children.items():
            score= ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
    def expanded(self):
        return len(self.children) > 0

    
    def select_actions(self,temperature):
        visit_count=np.array([child.visit_count for child in self.children.values()])
        actions= np.array(action for action in self.children.keys())
        if temperature == 0:
            return actions[np.argmax(visit_count)]
        else if temperature == float('inf'):
            return np.random.choice(actions)
        else:
            visit_count_distribution= visit_count ** (1 / temperature)
            visit_count_distribution /= visit_count_distribution.sum()
            return np.random.choice(actions, p=visit_count_distribution)   #return a random action based on the visit count distribution
        return action
    



def check_winner(board):
    # Rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != 0:
            return board[row][0] 
    # Cols
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return board[0][col] 
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0] 
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0


def is_full(board):
    return all(cell != 0 for row in board for cell in row)

# write ucb score here

def return_all_legal_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if check_legal_move(board, i, j)]




class MCTS:
    def __init__(self,game,model,args):
        self.game=game
        self.model=model
        self.args=args

    
    def run(self,model,state,to_play):
        root = Node(0,to_play)

        actions,values=model.predict(state)
        valid_moves=game.get_valid_moves(state)
        