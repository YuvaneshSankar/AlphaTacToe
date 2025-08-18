import torch
import math
import numpy as np


c_puct = 1.0

def ucb_score(parent,child):
    exploration=c_puct*child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    if child.visit_count >0:
        exploitation=-child.value()
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
        actions= np.array([action for action in self.children.keys()])
        if temperature == 0:
            return actions[np.argmax(visit_count)]
        elif temperature == float('inf'):
            return np.random.choice(actions)
        else:
            visit_count_distribution= visit_count ** (1 / temperature)
            visit_count_distribution /= visit_count_distribution.sum()
            return np.random.choice(actions, p=visit_count_distribution)   #return a random action based on the visit count distribution
    







#what am i trying to get
#from the game -> get valid moves  ,  generate new board
#from the model -> actions ,value
#from the args ->Number of sim

class MCTS:
    def __init__(self,game,model,args):
        self.game=game
        self.model=model
        self.args=args

    
    def run(self,model,state,to_play):
        root = Node(0,to_play)


        #EXPAND ROOT NODE

        actions,values=model.predict(state)   #action_probs = [0.2, 0.1, 0.3, 0.4] say we get this 
        valid_moves=self.game.get_valid_moves(state)  # valid_moves = [1, 0, 1, 0] 
        #now we have to mask the illegal moves
        action_probs=actions*valid_moves #After masking → [0.2, 0, 0.3, 0]
        #Now we have to normalize it to get prob sum=1
        denom = np.sum(action_probs)
        if denom == 0:
            action_probs = np.array(valid_moves, dtype=float) / max(1, np.sum(valid_moves))
        else:
            action_probs /= denom #[0.4, 0, 0.6, 0]
        root.expand(state,to_play,action_probs)

        for _ in range(self.args['number_simulation']):
            node=root
            search_path=[node]
            #if it is not a leaf node just select child and move to the selected node
            #so this goes till the leaf node trying to just append the best node everytime
            while node.expanded():
                action,node=node.select_child()
                search_path.append(node)
                
            #now we have reached a leaf node
            #so get the parent node of the leaf node 
            parent=search_path[-2]
            state=parent.state

            #Now we are going to get the next state if the player 1 takes a action now
            next_state,_=self.game.get_next_state(state,player=1,action=action)

            #now we have to see from the other person's view
            next_state=self.game.get_canonical_board(next_state,player=-1)

            #Now lets check whether the game is over or not
            value=self.game.get_reward_for_player(next_state,player=1)

            #so now its like we reached end player 1 takes a action then we go to player 2 pov and check the game terminal status

            if value is None:
                #meaning the game is not over so we have to expand
                actions,value=model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = actions * valid_moves 
                denom = np.sum(action_probs)
                if denom == 0:
                    action_probs = np.array(valid_moves, dtype=float) / max(1, np.sum(valid_moves))
                else:
                    action_probs /= denom
                node.expand(next_state, parent.to_play * -1, action_probs)
                #hparent.to_play = who’s turn it was in the parent. parent.to_play * -1 = switch perspective (because the child node is after the move)
                #but at the root we already know who it is player to_play we didnt do any moves to reach root state we start from there 

            #we send the predicted value from the model
            self.backpropagate(search_path,value,parent.to_play * -1)
        return root
    

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            #here if the current player from the expanded node earlier is the one in the node that we are backpropogating then add thier value
            #if it was the other guy then subract the value 
            node.value_sum+=value if node.to_play==to_play else -value
            node.visit_count += 1
