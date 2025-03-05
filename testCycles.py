import Cycles
import MCTS
import numpy
import ResNet
import ResNetLinear
import ResNetCycles
import torch
import RandomPlayer
import LineCycleMaker
import ConnectFour
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

# game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
game = ConnectFour.ConnectFour()
player = 1
for zx in range(16):
    # if zx != 4 and zx != 8 and zx != 16 and zx != 31 :
    #     continue
    args1 = {
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
        'directory': "./Data/BayesianModels/36",
        'trained_model': './Data/Manual/C/model_'+str(zx)+f'_{game}_ResNetCycles.pt'
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

    for z in range(1):
   #     if z%100==0:
   #         print(z)
        state = game.get_intial_state()
        print(state)
        while True:
            # print(state)
            # print()
            
            if player==1:
                policy, _ = model1(
                    torch.tensor(game.get_encoded_state(state),device=device).unsqueeze(0)
                )
                print(policy)
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
                policy, _ = model1(
                    torch.tensor(game.get_encoded_state(neutral_state),device=device).unsqueeze(0)
                )
                print(policy)
                mcts_probs = mcts1.search(neutral_state)
                #choosing the largest prob action
                action = numpy.argmax(mcts_probs)
            print(action, " taken")
            state = game.get_next_state(state,action,player)
            print(state)
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