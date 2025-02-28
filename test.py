import torch
import numpy
import Cycles
import LineCycleMaker
import IdealLinePlayer
import objectiveFunctionRandom
import BayesianOptimization



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
game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
# args = {'lr': (0.1913), 'weight_decay': (0.9423), 'num_resBlocks': 2, 'num_hidden': 16, 'C': (2.3832), 'num_searches': 10, 'num_selfPlay_iterations': 116, 'num_epochs': 3, 'batch_size': 20, 'temperature': (7.4322), 'dirichlet_epsilon': (0.9403), 'dirichlet_alpha': (0.1353), 'num_parallel_games': 80, 'num_iterations': 3, 'check_ai': True, 'directory': './Data/BayesianModels/0'}


args = {'lr': (0.0159), 'weight_decay': (0.8363), 'num_resBlocks': 1, 'num_hidden': 28, 'C': (0.9153), 'num_searches': 15, 'num_selfPlay_iterations': 109, 'num_epochs': 1, 'batch_size': 28, 'temperature': (3.7989), 'dirichlet_epsilon': (0.4572), 'dirichlet_alpha': (0.7367), 'num_parallel_games': 116, 'num_iterations': 3, 'check_ai': True, 'directory': './Data/BayesianModels/0'}
bounds = [
    [0.0001,1,0], # lr
    [0.00001,1,0], # weight_decay
    [1,3,1], # num_resBlocks
    [16,32,1], # num_hidden
    [0.5,6,0], # C
    [10,20,1], # num_searches
    [64,128,1], # num_selfPlay_iterations
    [1,4,1], # num_epochs
    [16,32,1], # batch_size
    [1,10,0], # temperature
    [0,1,0], # dirichlet_epsilon
    [0,1,0], # dirichlet_alpha
    [64,128,1] # num_parallel_games
]

def unscaling(args, bounds):
    newArgs = []
    for i, arg in enumerate(args):
        l = bounds[i][0]; u =bounds[i][1]
        if(bounds[i][2]==1):
            newArgs.append(int((u-l)*(arg)+l))
        else:
            newArgs.append((u-l)*(arg)+l)    
    return newArgs

t = torch.rand(13)
t = unscaling(t.tolist(), bounds=bounds)
print(t)

# newValue = BayesianOptimization.objFunction(game=game,args1=args, victoryCutoff=0.5)
# print(newValue)