import TicTacToe
import MCTS
import numpy
import ResNet
import ResNetCycles
import ResNetLinear
import AlphaZero
import AlphaZeroParallel
import torch
import torch.nn
import torch.nn.functional
import Cycles
import ConnectFour
import LineCycleMaker
from tqdm import trange

#game = TicTacToe.TicTacToe()
#game = ConnectFour.ConnectFour()

# adj_matrix = numpy.array([
#     [0,1,1,1],
#     [1,0,1,0],
#     [1,1,0,1],
#     [1,0,1,0]
# ])
# valid_cycles = numpy.array([
#     [0,1,2],
#     [3,2,0]
# ])

#house
adj_matrix = numpy.array( [
    [0,1,0,0,0,0,1],
    [1,0,1,0,0,0,1],
    [0,1,0,1,0,1,0],
    [0,0,1,0,1,0,0],
    [0,0,0,1,0,1,0],
    [0,0,1,0,1,0,1],
    [1,1,0,0,0,1,0]
] )
valid_cycles = [
    [0,1,6],
    [6,1,2,5],
    [5,2,3,4]
]

#line
# adj_matrix, valid_cycles = LineCycleMaker.LineGraph(6)

game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
# game = TicTacToe.TicTacToe()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
# args:  [0.7224303919732571, 0.2754741642761231, 6, 67, 1.6668683290481567, 110, 295, 9, 23, 3.5337294340133667, 0.10986435413360596, 0.20370936393737793, 126]
args = {
    'lr':0.7224303919732571,
    'weight_decay':0.2754741642761231,
    'num_resBlocks': 6,
    'num_hidden': 67,
    'C' : 1.6668683290481567,
    'num_searches': 110,
    'num_iterations': 16,
    'num_selfPlay_iterations': 295,
    'num_epochs': 9,
    'batch_size': 23,
    'temperature' : 3.5337294340133667,
    'dirichlet_epsilon': 0.10986435413360596,
    'dirichlet_alpha': 0.20370936393737793,
    'num_parallel_games': 126,
    'check_ai':True,
    'directory': "./Data/Manual/A"
    # 'directory': "/Users/vigyansahai/Code/AlphaZeroCopy/Data/C"
}

# model = ResNet.ResNet(game, args['num_resBlocks'], args['num_hidden'], device)
model = ResNetCycles.ResNetCycles(game, args['num_resBlocks'], args['num_hidden'], device)
# model = ResNetLinear.ResNetLinear(game, args['num_resBlocks'], args['num_hidden'], device)
model = model.to(device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args['lr'], max_lr=args['max_lr'],step_size_up=4, step_size_down=4,mode="triangular", cycle_momentum=False)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,11,14], gamma=0.1, )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=16)
#alpha = AlphaZero.AlphaZero(model, optimizer, game, args)
alpha = AlphaZeroParallel.AlphaZeroParallel(model, optimizer, game, args, scheduler)
# alpha = AlphaZeroParallel.AlphaZeroParallel(model, optimizer, game, args, scheduler)

alpha.learn()