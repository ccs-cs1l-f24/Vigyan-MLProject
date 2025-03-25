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
matchesP1 = numpy.zeros((16,16))
matchesP2 = numpy.zeros((16,16))
for zy in range(16):
    for zx in range(16):
        print(zx,zy)
        # if(zx!=15): continue
        args1 = {
            'lr':0.14653052592277527,
            'weight_decay':0.060361624289155015,
            'num_resBlocks': 6,
            'num_hidden': 88,
            'C' : 4.1289102435112,
            'num_searches': 125,
            'num_iterations': 16,
            'num_selfPlay_iterations': 734,
            'num_epochs': 3,
            'batch_size': 30,
            'temperature' : 1.7032934427261353,
            'dirichlet_epsilon': 0.2207707166671753,
            'dirichlet_alpha': 0.13379442691802979,
            'num_parallel_games': 128,
            'check_ai':True,
            'directory': "./Data/Manual/A",
            'trained_model': './Data/BayesianModels/55/model_'+str(zy)+f'_{game}_ResNetCycles.pt'
        }
        
        args2 = {
            'lr':0.17831012549006348,
            'weight_decay':0.06814283855740606,
            'num_resBlocks': 10,
            'num_hidden': 97,
            'C' : 1.7073375876688646,
            'num_searches': 91,
            'num_iterations': 16,
            'num_selfPlay_iterations': 914,
            'num_epochs': 2,
            'batch_size': 47,
            'temperature' : 6.890213918474457,
            'dirichlet_epsilon': 0.6914332406429372,
            'dirichlet_alpha': 0.37384641938285845,
            'num_parallel_games': 128,
            'check_ai':True,
            'directory': "./Data/Manual/A",
            'trained_model': './Data/BayesianModels/69/model_'+str(zx)+f'_{game}_ResNetCycles.pt'
        }
    

        args1['dirichlet_epsilon']=0
        args2['dirichlet_epsilon']=0

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

        
        for first in range(-1,2,2):
            win = 0
            lose = 0
            player = 1
            # flag = True
            print('first: ',first)
            for z in range(5):
        #     if z%100==0:
        #         print(z)
                state = game.get_intial_state()

                while True:
        #            print(state)
                    
                    if player==first:
                        # print('2')
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
                        # print('1')
                        #Monty
                        neutral_state = game.change_perspective(state, player)
                        mcts_probs = mcts1.search(neutral_state)
                        #choosing the largest prob action
                        action = numpy.argmax(mcts_probs)

                        # rp = RandomPlayer.RandomPlayer()
                        # action = rp.action(game.get_valid_moves(state))

                    state = game.get_next_state(state,action,player)
                    # if(flag):
                    #     print(state)
                    #     flag = False

                    value, is_terminal = game.get_value_and_terminate(state,action)

                    if is_terminal:
        #               
                        if player==1:
                            win = win+1
                        else:
                            lose = lose+1
                        break

                    player = game.get_opponent(player)
            if(first==-1):
                print("win1: ", win, " win2: ", lose )
                matchesP1[zx][zy] +=win
                matchesP2[zx][zy] +=lose
            if(first==1):
                print("win1: ", lose, " win2: ", win )
                matchesP1[zx][zy] +=lose
                matchesP2[zx][zy] +=win
            
    print()
print(matchesP1)
print(matchesP2)


# args1 = {
#             'lr':0.14653052592277527,
#             'weight_decay':0.060361624289155015,
#             'num_resBlocks': 6,
#             'num_hidden': 88,
#             'C' : 4.1289102435112,
#             'num_searches': 125,
#             'num_iterations': 16,
#             'num_selfPlay_iterations': 734,
#             'num_epochs': 3,
#             'batch_size': 30,
#             'temperature' : 1.7032934427261353,
#             'dirichlet_epsilon': 0.2207707166671753,
#             'dirichlet_alpha': 0.13379442691802979,
#             'num_parallel_games': 128,
#             'check_ai':True,
#             'directory': "./Data/Manual/A",
#             'trained_model': './Data/Manual/B/model_'+str(zx)+f'_{game}_ResNetCycles.pt'
#         }

#         args2 = {
#             'lr':0.14653052592277527,
#             'weight_decay':0.060361624289155015,
#             'num_resBlocks': 6,
#             'num_hidden': 88,
#             'C' : 4.1289102435112,
#             'num_searches': 125,
#             'num_iterations': 16,
#             'num_selfPlay_iterations': 734,
#             'num_epochs': 3,
#             'batch_size': 30,
#             'temperature' : 1.7032934427261353,
#             'dirichlet_epsilon': 0.2207707166671753,
#             'dirichlet_alpha': 0.13379442691802979,
#             'num_parallel_games': 128,
#             'check_ai':True,
#             'directory': "./Data/Manual/A",
#             'trained_model': './Data/Manual/A/model_'+str(zy)+f'_{game}_ResNetCycles.pt'
#         }
