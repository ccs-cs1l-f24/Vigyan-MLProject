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
            
        

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        
        #This section adds noise to the state's inital moves, not sure why we are doing it here
        #tho, and not sure how it works, need to look at the hyper stuff
        # _ is a blank variable, since we dont use it
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states),device=self.model.device)
        )
        #no squeeze here because we want to keep the batches
        policy = torch.softmax(policy, axis = 1).cpu().numpy()
        #Dirichlet hyper para, remember \ is a newline
        policy = (1 - self.args['dirichlet_epsilon'])*policy + (self.args['dirichlet_epsilon']) \
            * numpy.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
            #we format the noise to be of the same dimension as each specific batch, not every batch, 
            # This ensures that every batch gets the same noise map
            # NOTE, REVISIT THIS LATER TO UNDERSTAND BETTER!!!!
        
        #Looping through every batch
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves #FREAKING GENIUS, any illigal moves are 0
            spg_policy /= numpy.sum(spg_policy)
            
            
            spg.root = Node(self.game, self.args, states[i], visit_count = 1) 
            spg.root.expand(spg_policy)
        

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root
                # selection

                while(node.is_fully_expanded()):
                    node = node.select()
                    
                
                value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node
                
            #All the games not None, creates a list of indecies
            expandable_spGames = [i for i in range(len(spGames)) if spGames[i].node is not None]
                
            if len(expandable_spGames) > 0:
                #recreates states to only have the expandable states
                states = numpy.stack([spGames[i].node.state for i in expandable_spGames])
                
                policy, value = self.model(
                    #no need to unqueeze since it already has another dim from batching
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                #softmax at 1 since pos 0 is batch index
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                
                value = value.cpu().numpy()
                
            #mappingIdx = expandable_spGames[i] 
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                # print(spg_policy)
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves #FREAKING GENIUS, any illigal moves are 0
                if(numpy.sum(spg_policy)==0):
                    spg_policy = numpy.ones(len(valid_moves)) * valid_moves
                    # breakpoint()
                    
                spg_policy /= numpy.sum(spg_policy)
                
                node.expand(spg_policy)
                node.backpropagate(spg_value)
