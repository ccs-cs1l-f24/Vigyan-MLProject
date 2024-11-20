import TicTacToe
import MCTS
import numpy
import ResNet
import torch
import ConnectFour

game = ConnectFour.ConnectFour()
player = 1

args = {
    'C' : 2,
    'num_searches': 60,
    'dirichlet_epsilon': 0,
    'dirichlet_alpha': 0.3,
    'trained_model': '/Users/vigyansahai/Code/AlphaZeroCopy/Data/model_2_ConnectFour.pt'
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ResNet.ResNet(game, 9, 128, device=device)
model.load_state_dict(torch.load(args['trained_model'], map_location=device))
model.eval()

mcts = MCTS.MCTS(game,args,model)


state = game.get_intial_state()

while True:
    print(state)
    print("\n")
    if player==1:
        valid_moves = game.get_valid_moves(state)
        print(valid_moves)
        action = int(input(f"{player}:"))

        if action>=len(valid_moves) or valid_moves[action]==0:
            print("not valid idot")
            continue
    else:
        #Monty
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        #choosing the largest prob action
        action = numpy.argmax(mcts_probs)

    state = game.get_next_state(state,action,player)

    value, is_terminal = game.get_value_and_terminate(state,action)

    if is_terminal:
        print(state)
        if value==1:
            print(player,"won")
        else:
            print("draw")
        break

    player = game.get_opponent(player)