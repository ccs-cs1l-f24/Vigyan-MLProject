import Cycles
import MCTS
import MCTSParallel
import AlphaZeroParallel
import numpy
import ResNet
import ResNetLinear
import ResNetCycles
import torch
import RandomPlayer
import LineCycleMaker
import BayesianOptimization


# the map:
#        (0)
#       / | \
#    (3)  |  (1)
#       \ | /
#        (2)

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

#line
# adj_matrix, valid_cycles = LineCycleMaker.LineGraph(6)

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

game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
changeArgs1 = [
    'lr',
    'weight_decay',
    'num_resBlocks',
    'num_hidden',
    'C',
    'num_searches',
    'num_selfPlay_iterations',
    'num_epochs',
    'batch_size',
    'temperature',
    'dirichlet_epsilon',
    'dirichlet_alpha'
]
staticArgs1 = {
    'num_iterations':16,
    'num_parallel_games':256,
    'check_ai':True,
    'directory':'./Data/BayesianModels/'
}

#bound [lower, upper) isInt
bounds = [
    [0.0001,0.5,0], # lr
    [0.00001,1,0], # weight_decay
    [7,14,1], # num_resBlocks
    [16,128,1], # num_hidden
    [0.5,10,0], # C
    [8,128,1], # num_searches
    [512,1024,1], # num_selfPlay_iterations
    [1,12,1], # num_epochs
    [16,64,1], # batch_size
    [1,10,0], # temperature
    [0,1,0], # dirichlet_epsilon
    [0,1,0], # dirichlet_alpha
]
def unscaling(args, bounds):
    newArgs = []
    for i, arg in enumerate(args):
        l = bounds[i][0]; u =bounds[i][1]
        if(bounds[i][2]==1):
            newArgs.append(int((u-l)*(arg)+l))
        else:
            newArgs.append(((u-l)*(arg)+l))    
    return newArgs
        

def scaling(args, bounds):
    newArgs = []
    for i, arg in enumerate(args):
        l = bounds[i][0]; u =bounds[i][1]
        newArgs.append((arg-l)/(u-l))
    return newArgs

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# bestRatio:  0.6875
# args:  [0.7224303919732571, 0.2754741642761231, 6, 67, 1.6668683290481567, 110, 295, 9, 23, 3.5337294340133667, 0.10986435413360596, 0.20370936393737793, 126]

args, value, index = BayesianOptimization.bayesian_opt\
                (game=game,iterations=64, numSamples=5000, setparamsDict=staticArgs1,\
                changeparams=changeArgs1, kernal=BayesianOptimization.Matern52, \
                scaling=scaling, unscaling=unscaling, bounds=bounds, victoryCutoff=0.7, \
                guesses=4, load_save=True
                )
                
print('bestRatio: ',value)
print('args: ', args)
print('index', index)



# bounds = [
#     [0.0001,1,0], # lr
#     [0.00001,1,0], # weight_decay
#     [1,8,1], # num_resBlocks
#     [16,128,1], # num_hidden
#     [0.5,10,0], # C
#     [8,128,1], # num_searches
#     [64,512,1], # num_selfPlay_iterations
#     [1,12,1], # num_epochs
#     [16,64,1], # batch_size
#     [1,10,0], # temperature
#     [0,1,0], # dirichlet_epsilon
#     [0,1,0], # dirichlet_alpha
#     [64,256,1] # num_parallel_games
# ]

# bounds = [
#     [0.0001,0.5,0], # lr
#     [0.00001,1,0], # weight_decay
#     [7,14,1], # num_resBlocks
#     [16,128,1], # num_hidden
#     [0.5,10,0], # C
#     [8,128,1], # num_searches
#     [512,1024,1], # num_selfPlay_iterations
#     [1,12,1], # num_epochs
#     [16,64,1], # batch_size
#     [1,10,0], # temperature
#     [0,1,0], # dirichlet_epsilon
#     [0,1,0], # dirichlet_alpha
# ]
