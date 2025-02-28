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

class SPG():
    def __init__(self, game):
        self.state = game.get_intial_state()
        self.root = None
        self.node =None
        


# arg must have directory for each Bayesian iteration
def objFunction(game, args1, victoryCutoff):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # AlphaZeroTest:
    
    # model = ResNet.ResNet(game, args['num_resBlocks'], args['num_hidden'], device)
    model = ResNetCycles.ResNetCycles(game, args1['num_resBlocks'], args1['num_hidden'], device)
    # model = ResNetLinear.ResNetLinear(game, args['num_resBlocks'], args['num_hidden'], device)
    model = model.to(device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args1['lr'], weight_decay=args1['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args1['num_iterations'])
    alpha = AlphaZeroParallel.AlphaZeroParallel(model, optimizer, game, args1, scheduler)
    alpha.learn()
    
    # Done with generating the models, we now chech each one
    
    totalwin = 0
    totallose = 0
    
    player = 1
    for zx in range(args1['num_iterations']):
        model_location =  args1['directory']+f"/model_{zx}_{game}_{model}.pt"

        model1 = ResNetCycles.ResNetCycles(game, args1['num_resBlocks'], args1['num_hidden'], device=device)
        rp = RandomPlayer.RandomPlayer()
        model1.load_state_dict(torch.load(model_location,map_location=device))
        model1.eval()
        mcts1 = MCTSParallel.MCTSParallel(game,args1,model1)

        win = 0
        lose = 0
        #Code from AlphaZeroParallel
        for first in range(-1,2,2):
            player = 1
            spGames = [ SPG(game) for spg in range(100)]
            while len(spGames) > 0:
                # first==-1, random goes 2nd, first==1, random goes 1st
                if player==first:
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
            if(zx>(victoryCutoff)*(args1['num_iterations'])):
                totalwin = totalwin + win
                totallose = totallose + lose
        print()
    ratio = totalwin/(totalwin+totallose)
    print(totalwin, totalwin+totallose, ratio)
    return ratio
    