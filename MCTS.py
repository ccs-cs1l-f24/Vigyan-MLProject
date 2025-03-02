import TicTacToe
import ResNet
import numpy
import math
import torch

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior_input = 0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior_input = prior_input

        self.children = []
        #self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = visit_count
        self.value_sum =  0

    def is_fully_expanded(self):
        # return numpy.sum(self.expandable_moves) ==0 and len(self.children) >0
        return len(self.children) >0
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
        if(child.visit_count==0):
            q_value=0
        else:
            #flip the q_value to get smallest
            q_value = 1- ((child.value_sum / child.visit_count) +1)/2
        
        return q_value + (self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count+1)) * child.prior_input)
        # return q_value + (self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count))
        
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob>0:
                # action = numpy.random.choice(numpy.where(self.expandable_moves==1)[0])
                # self.expandable_moves[action] = 0
                child_state = self.state.copy()
                #the player=1 since every node thinks its player one but percieves the other nodes as opposite
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state,player = -1)
                
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
        # return child
    
    # def simulate(self):
    #     value, is_terminal = self.game.get_value_and_terminate(self.state,self.action_taken)
    #     value = self.game.get_opponent_value(value)
        
    #     if is_terminal:
    #         return value
        
    #     rollout_state = self.state.copy()
    #     rollout_player = 1
    #     while True:
    #         valid_move = self.game.get_valid_moves(rollout_state)
    #         action = numpy.random.choice(numpy.where(valid_move==1)[0])
    #         rollout_state = self.game.get_next_state(rollout_state,action, rollout_player)
    #         value, is_terminal = self.game.get_value_and_terminate(rollout_state, action)
    #         if is_terminal:
    #             if rollout_player == -1:
    #                 value = self.game.get_opponent_value(value)
    #             return value
            
    #         rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum+=value
        self.visit_count+=1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)
            
        

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count = 1) #head does not need the other values
        
        #This section adds noise to the state's inital moves, not sure why we are doing it here
        #tho, and not sure how it works, need to look at the hyper stuff
        # _ is a blank variable, since we dont use it
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state),device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis = 1).squeeze(0).cpu().numpy()
        #Dirichlet hyper para
        policy = (1 - self.args['dirichlet_epsilon'])*policy + (self.args['dirichlet_epsilon']) \
            * numpy.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves #FREAKING GENIUS, any illigal moves are 0
        if(numpy.sum(policy)==0):
            policy = valid_moves
            # breakpoint()
        policy /= numpy.sum(policy)
        root.expand(policy)
        

        for search in range(self.args['num_searches']):
            node = root
            # selection

            while(node.is_fully_expanded()):
                node = node.select()
                
            
            value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                # a = torch.tensor(self.game.get_encoded_state(node.state))
                # print("size: ")
                # print(a.size())
                # a = a.unsqueeze(0)
                # print(a.size())
                policy, value = self.model(
                    #in the form of the 3 layers, turn it into a tensor from numpy stuff
                    #do unsqueeze at pos 0 to add an extra dim for the flatten command in
                    #ResNet to work
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                #softmax at 1 since pos 0 is the dummy dim, squeeze at 0 to get rid of dummy dim
                #.cpu() moves the tensor to the cpu, since it could be on the gpu
                #numpy only works on cpu, now we turn it back to numpy
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves #FREAKING GENIUS, any illigal moves are 0
                policy /= numpy.sum(policy)
                
                value = value.item()
                
                # expansion
                node.expand(policy)
                #node = node.expand()
                #simulation
                #value = node.simulate() #NOT ANYMORE, ai better
            
            # backprop
            node.backpropagate(value)
        
        # return visit_counts
        action_probs = numpy.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= numpy.sum(action_probs)
        return action_probs