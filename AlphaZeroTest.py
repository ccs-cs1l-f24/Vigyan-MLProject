import TicTacToe
import MCTS
import numpy
import ResNet
import AlphaZero
import torch
import torch.nn
import torch.nn.functional
from tqdm import trange

game = TicTacToe.TicTacToe()

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ResNet.ResNet(game, 4, 64)# device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C' : 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature' : 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alpha = AlphaZero.AlphaZero(model, optimizer, game, args)
alpha.learn()