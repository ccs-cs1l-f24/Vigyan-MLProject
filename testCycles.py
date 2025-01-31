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
player = 1
for zx in range(16):
    # if zx != 4 and zx != 8 and zx != 16 and zx != 31 :
    #     continue
    args1 = {
        'lr':0.002,
        'weight_decay':0.0001,
        'num_resBlocks': 10,
        'num_hidden': 64,
        'C' : 4,
        'num_searches': 50,
        'num_iterations': 16,
        'num_selfPlay_iterations': 240,
        'num_epochs': 8,
        'batch_size': 20,
        'temperature' : 1,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.1,
        'num_parallel_games': 120,
        'check_ai':True,
        'trained_model': '/Users/vigyansahai/Code/AlphaZeroCopy/Data/B/model_'+str(zx)+'_Cycles_ResNetCycles.pt'
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # model1 = ResNet.ResNet(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    # model1 = ResNetLinear.ResNetLinear(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    model1 = ResNetCycles.ResNetCycles(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    
    model1.load_state_dict(torch.load(args1['trained_model'],map_location=device))
    
    model1.eval()

    mcts1 = MCTS.MCTS(game,args1,model1)

    win = 0
    lose = 0

    for z in range(100):
   #     if z%100==0:
   #         print(z)
        state = game.get_intial_state()

        while True:
            # print(state)
            # print()
            
            if player==-1:
                valid_moves = game.get_valid_moves(state)
                # print(valid_moves)
                
                rp = RandomPlayer.RandomPlayer()
                
                # action = int(input(f"{player}:"))
                action = rp.action(valid_moves)

                if valid_moves[action]==0:
                    print("not valid idot")
                    continue
            else:
                #Monty
                neutral_state = game.change_perspective(state, player)
                mcts_probs = mcts1.search(neutral_state)
                #choosing the largest prob action
                action = numpy.argmax(mcts_probs)

            state = game.get_next_state(state,action,player)

            value, is_terminal = game.get_value_and_terminate(state,action)

            if is_terminal:
                numpy.set_printoptions(linewidth=numpy.nan)
                # print(state)
                if value==1:
                    # print(player,"won")
                    if player==1:
                        win = win+1
                    else:
                        lose = lose+1
                else:
                    # print(player,"won")
                    if player==1:
                        win = win+1
                    else:
                        lose = lose+1
                break

            player = game.get_opponent(player)
    print("win1: ", win, " lose: ", lose )
    # f = open("WLfiles/Cycles_ResNetCycles.txt", "a")
    # f.write(str(win)+", "+str(lose)+"\n")
    # f.close()
print()