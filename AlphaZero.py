import Cycles
import MCTS
import numpy
import ResNet
import random
import torch
import torch.nn
import torch.nn.functional
from tqdm import trange

class AlphaZero:
    def __init__(self, model, optimizer, game: Cycles.Cycles, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS.MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_intial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))

            #this is a hyper-parameter to add randomness into the action chosen to play by the AI
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])

            action = numpy.random.choice(self.game.action_size, p=(temperature_action_probs / numpy.sum(temperature_action_probs)))
            
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminate(state, action)
            
            if is_terminal:
                return_memory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory
            
            else:
                player = self.game.get_opponent(player)
                    
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
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            #turn back on training
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            directory = self.args['directory']
            torch.save(self.model.state_dict(), directory+"/"+f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), directory+"/"+f"optimizer_{iteration}_{self.game}.pt")