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
# game = ConnectFour.ConnectFour()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
# args:  [0.7224303919732571, 0.2754741642761231, 6, 67, 1.6668683290481567, 110, 295, 9, 23, 3.5337294340133667, 0.10986435413360596, 0.20370936393737793, 126]

# args = {'lr': 0.9721780825912952, 'weight_decay': 0.1823386658537388, 'num_resBlocks': 11, 'num_hidden': 96, 'C': 3.9406969249248505, 'num_searches': 76, 'num_selfPlay_iterations': 666, 'num_epochs': 1, 'batch_size': 22, 'temperature': 5.730639934539795, 'dirichlet_epsilon': 0.055349528789520264, 'dirichlet_alpha': 0.7023180723190308, 'num_iterations': 16, 'num_parallel_games': 256, 'check_ai': True, 'directory': './Data/Manual/B'}

# args:  [0.04198784477710724, 0.57606262717247, 9, 58, 2.7265064418315887, 44, 973, 6, 37, 3.9412047266960144, 0.238503098487854, 0.05893164873123169]
args = {
    'lr':0.002,
    'weight_decay':0.0001,
    'num_resBlocks': 9,
    'num_hidden': 58,
    'C' : 2.7265064418315887,
    'num_searches': 44,
    'num_iterations': 16,
    'num_selfPlay_iterations': 973,
    'num_epochs': 6,
    'batch_size': 37,
    'temperature' : 3.9412047266960144,
    'dirichlet_epsilon': 0.238503098487854,
    'dirichlet_alpha': 0.05893164873123169,
    'num_parallel_games': 128,
    'check_ai':True,
    'directory': "./Data/Manual/D"
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
