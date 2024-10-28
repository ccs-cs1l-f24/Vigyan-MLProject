import ResNet
import TicTacToe
import MCTS
import torch
import matplotlib.pyplot as plt

game = TicTacToe.TicTacToe()

state = game.get_intial_state()

state = game.get_next_state(state, 2, -1)
state = game.get_next_state(state, 4, -1)
state = game.get_next_state(state, 6, 1)
state = game.get_next_state(state, 8, 1)

print(state)

encoded_state = game.get_encoded_state(state)

print(encoded_state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

model = ResNet.ResNet(game, 4, 64)
model.load_state_dict(torch.load('/Users/vigyansahai/Code/AlphaZeroCopy/Data/model_2.pt'))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value, policy)

plt.bar(range(game.action_size),policy)
plt.show()