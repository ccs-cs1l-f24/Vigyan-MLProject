import TicTacToe
import numpy
import math

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum =  0

    def is_fully_expanded(self):
        return numpy.sum(self.expandable_moves) ==0 and len(self.children) >0
    def select(self):
        best_child = None
        best_ucb = -numpy.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child
    
    def get_ucb(self, child):
        #flip the q_value to get smallest
        q_value = 1- ((child.value_sum / child.visit_count) +1)/2
        
        return q_value + (self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count))
        
    def expand(self):
        action = numpy.random.choice(numpy.where(self.expandable_moves==1)[0])
        self.expandable_moves[action] = 0
        
        child_state = self.state.copy()
        #the player=1 since every node thinks its player one but percieves the other nodes as opposite
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state,player = -1)
        
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminate(self.state,self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_move = self.game.get_valid_moves(rollout_state)
            action = numpy.random.choice(numpy.where(valid_move==1)[0])
            rollout_state = self.game.get_next_state(rollout_state,action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminate(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            
            rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum+=value
        self.visit_count+=1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)
            
        

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state, )

        for search in range(self.args['num_searches']):
            node = root
            # selection

            while(node.is_fully_expanded()):
                node = node.select()
            
            value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                # expansion
                node = node.expand()
                #simulation
                value = node.simulate()
            
            # backprop
            node.backpropagate(value)
        
        # return visit_counts
        action_probs = numpy.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= numpy.sum(action_probs)
        return action_probs