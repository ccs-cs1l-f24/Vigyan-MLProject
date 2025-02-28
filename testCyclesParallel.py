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

class SPG():
    def __init__(self, game):
        self.state = game.get_intial_state()
        self.root = None
        self.node =None

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
        'directory': "./Data/Manual/A",
        'trained_model': './Data/Manual/A/model_'+str(zx)+'_Cycles_ResNetCycles.pt'
    }
    args1['dirichlet_epsilon']=0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # model1 = ResNet.ResNet(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    # model1 = ResNetLinear.ResNetLinear(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    model1 = ResNetCycles.ResNetCycles(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
    rp = RandomPlayer.RandomPlayer()
    model1.load_state_dict(torch.load(args1['trained_model'],map_location=device))
    
    model1.eval()

    mcts1 = MCTSParallel.MCTSParallel(game,args1,model1)

    win = 0
    lose = 0
    #Code from AlphaZeroParallel
    player = 1
    spGames = [ SPG(game) for spg in range(100)]
    while len(spGames) > 0:
        if player==1:
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                if(numpy.sum(game.get_valid_moves(spg.state))==0): print('da hec',i)
                valid_moves = game.get_valid_moves(spg.state)
                if(numpy.sum(valid_moves)==0): print('da hec',i)
                # else: print('size',numpy.sum(valid_moves))
                if(valid_moves.size==0): print(':(')
                # print(valid_moves)
                action = rp.action(valid_moves)
                
                spg.state = game.get_next_state(spg.state, action, player)
                
                value, is_terminal = game.get_value_and_terminate(spg.state, action)
                # print(i)
                if is_terminal:
                    # print('killed',i)
                    # numpy.set_printoptions(linewidth=numpy.nan)
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
                    del spGames[i]
            player = game.get_opponent(player)
            continue
        states = numpy.stack([spg.state for spg in spGames])
        neutral_states = game.change_perspective(states, player)
        mcts1.search(neutral_states, spGames)
        
        #the [::-1] flips the index to go backwards to avoid array size issues later
        for i in range(len(spGames))[::-1]:
            spg = spGames[i]
            if(numpy.sum(game.get_valid_moves(spg.state))==0): print('da hecc')
            # taken from MCTS
            action_probs = numpy.zeros(game.action_size)
            for child in spg.root.children:
                action_probs[child.action_taken] = child.visit_count
            flag = False
            if(numpy.sum(action_probs)==0): 
                # print('ap: ',action_probs)
                flag = True
            action_probs /= numpy.sum(action_probs)

            #this is a hyper-parameter to add randomness into the action chosen to play by the AI
            temperature_action_probs = action_probs ** (1 / args1['temperature'])
            # if(flag): print('tap: ',temperature_action_probs)
            # action = numpy.random.choice(game.action_size, p=(temperature_action_probs / numpy.sum(temperature_action_probs)))
            # print('ai: ',game.get_valid_moves(spg.state))
            action = numpy.argmax(temperature_action_probs)
            
            spg.state = game.get_next_state(spg.state, action, player)
            
            value, is_terminal = game.get_value_and_terminate(spg.state, action)
            # print('ai',i)
            if is_terminal:
                # print('AIkilled',i)
                # numpy.set_printoptions(linewidth=numpy.nan)
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
                del spGames[i]
        player = game.get_opponent(player)
    print("win1: ", win, " lose: ", lose )
print()
        
        
        

