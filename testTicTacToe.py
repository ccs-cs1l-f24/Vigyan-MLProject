import TicTacToe
import MCTS
import numpy

game = TicTacToe.TicTacToe()
player = 1

args = {
    'C': 1.41,
    'num_searches': 100
}

mcts = MCTS.MCTS(game,args)


state = game.get_intial_state()

while True:
    print(state)
    if player==1:
        valid_moves = game.get_valid_moves(state)
        print(valid_moves)
        action = int(input(f"{player}:"))

        if valid_moves[action]==0:
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