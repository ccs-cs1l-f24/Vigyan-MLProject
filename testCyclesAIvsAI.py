import Cycles
import MCTS
import numpy
import ResNet
import ResNetLinear
import ResNetCycles
import torch
import RandomPlayer
import LineCycleMaker
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
# game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)

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

player = 1
# print("1 is rowcol, 2 is fully connected")
for zx in range(16):
    args1 = {
        'lr':0.002,
        'weight_decay':0.0001,
        'num_resBlocks': 7,
        'num_hidden': 64,
        'C' : 2,
        'num_searches': 100,
        'num_iterations': 16,
        'num_selfPlay_iterations': 256,
        'num_epochs': 4,
        'batch_size': 32,
        'temperature' : 1,
        'dirichlet_epsilon': 0.7,
        'dirichlet_alpha': 0.3,
        'num_parallel_games': 128,
        'check_ai':True,
        'trained_model': './Data/Manual/E-Control/model_'+str(zx)+'_Cycles_ResNetCycles.pt'
    }

    args2 = {
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
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.20370936393737793,
        'num_parallel_games': 126,
        'check_ai':True,
        'trained_model': './Data/Manual/A/model_'+str(zx)+'_Cycles_ResNetCycles.pt'
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # model = ResNet.ResNet(game, args['num_resBlocks'], args['num_hidden'], device=device)
    model1 = ResNetCycles.ResNetCycles(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    model2 = ResNetCycles.ResNetCycles(game, args2['num_resBlocks'], args2['num_hidden'], device=device)
    # model2 = ResNetLinear.ResNetLinear(game, args2['num_resBlocks'], args2['num_hidden'], device=device)
    model1.load_state_dict(torch.load(args1['trained_model'],map_location=device))
    model2.load_state_dict(torch.load(args2['trained_model'],map_location=device))
    model1.eval()
    model2.eval()

    mcts1 = MCTS.MCTS(game,args1,model1)
    mcts2 = MCTS.MCTS(game,args2,model2)

    win = 0
    lose = 0

    for z in range(5):
   #     if z%100==0:
   #         print(z)
        state = game.get_intial_state()

        while True:
#            print(state)
            
            if player==1:
                #args2
                neutral_state = game.change_perspective(state, player)
                mcts_probs = mcts2.search(neutral_state)
                #choosing the largest prob action
                action = numpy.argmax(mcts_probs)
                
                
                # valid_moves = game.get_valid_moves(state)
  #              print(valid_moves)
                
                # rp = RandomPlayer.RandomPlayer()
                
                # action = int(input(f"{player}:"))
                # action = rp.action(valid_moves)

                # if valid_moves[action]==0:
  #                  print("not valid idot")
                    # continue
            else:
                #Monty
                neutral_state = game.change_perspective(state, player)
                mcts_probs = mcts1.search(neutral_state)
                #choosing the largest prob action
                action = numpy.argmax(mcts_probs)

            state = game.get_next_state(state,action,player)

            value, is_terminal = game.get_value_and_terminate(state,action)

            if is_terminal:
 #               print(state)
                if value==1:
   #                 print(player,"won")
                    if player==1:
                        win = win+1
                    else:
                        lose = lose+1
                else:
    #                print(player,"won")
                    if player==1:
                        win = win+1
                    else:
                        lose = lose+1
                break

            player = game.get_opponent(player)
    print("win1: ", win, " win2: ", lose )
print()