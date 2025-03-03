import TicTacToe
import MCTS
import numpy
import ResNet
import AlphaZero
import AlphaZeroParallel
import torch
import torch.nn
import torch.nn.functional
import Cycles
import ConnectFour
from tqdm import trange

#game = TicTacToe.TicTacToe()
#game = ConnectFour.ConnectFour()

adj_matrix = numpy.array([
    [0,1,1,1],
    [1,0,1,0],
    [1,1,0,1],
    [1,0,1,0]
])
valid_cycles = numpy.array([
    [0,1,2],
    [3,2,0]
])
# game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
game = TicTacToe.TicTacToe()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")

args = {
    'num_resBlocks': 2,
    'num_hidden': 16,
    'C' : 2,
    'num_searches': 64,
    'num_iterations': 4,
    'num_selfPlay_iterations': 64,
    'num_epochs': 4,
    'batch_size': 32,
    'temperature' : 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'num_parallel_games': 16,
    'directory': "/Users/vigyansahai/Code/AlphaZeroCopy/Data"
}

model = ResNet.ResNet(game, args['num_resBlocks'], args['num_hidden'], device)
model = model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

#alpha = AlphaZero.AlphaZero(model, optimizer, game, args)
alpha = AlphaZeroParallel.AlphaZeroParallel(model, optimizer, game, args)

alpha.learn()