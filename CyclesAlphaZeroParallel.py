import Cycles
import MCTSParallel
import numpy
import ResNet
import random
import torch
import torch.nn
import torch.nn.functional
from tqdm import trange

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game: Cycles.Cycles, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel.MCTSParallel(game, args, model)
        
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = numpy.stack([spg.state for spg in spGames])
             
            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)
            
            #the [::-1] flips the index to go backwards to avoid array size issues later
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                # taken from MCTS
                action_probs = numpy.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= numpy.sum(action_probs)
                
                spg.memory.append((spg.root.state, action_probs, player))

                #this is a hyper-parameter to add randomness into the action chosen to play by the AI
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])

                action = numpy.random.choice(self.game.action_size, p=(temperature_action_probs / numpy.sum(temperature_action_probs)))
                
                spg.state = self.game.get_next_state(spg.state, action, player)
                
                value, is_terminal = self.game.get_value_and_terminate(spg.state, action)
                
                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    #this deletion is ok since we are looping backwards
                    del spGames[i]
                    
            player = self.game.get_opponent(player)
            
        return return_memory
                    
    def train(self, memory):
        random.shuffle(memory)
        for batchIndex in range (0,len(memory), self.args['batch_size']):
            sample = memory[batchIndex:min(len(memory)-1,batchIndex+self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state = numpy.array(state)
            policy_targets = numpy.array(policy_targets)
            value_targets = numpy.array(value_targets).reshape(-1,1)
            
            #tensorinnnggg
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            policy_loss = torch.nn.functional.cross_entropy(out_policy, policy_targets)
            value_loss = torch.nn.functional.mse_loss(out_value,value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            
        
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            #training data for one cycle
            memory = []
            
            #turn off parts of model for training to play fast
            self.model.eval()
            #using trange to show progress bar instead of range
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                
            #turn back on training
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            directory = "/Users/vigyansahai/Code/AlphaZeroCopy/Data"
            torch.save(self.model.state_dict(), directory+"/"+f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), directory+"/"+f"optimizer_{iteration}_{self.game}.pt")
            
class SPG():
    def __init__(self, game):
        self.state = game.get_intial_state()
        self.memory = []
        self.root = None
        self.node =None