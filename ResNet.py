import numpy
import math
import TicTacToe
import MCTS
import torch
import torch.nn
import torch.nn.functional

class ResNet(torch.nn.Module):
    numHIdden = 1
    def __init__(self, game, num_resBlocks, num_hidden):#, device):
        super().__init__()
        # self.device = device
        # self.numHIdden = num_hidden
        # self.ggame = game
        self.startBlock = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_hidden, kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(num_hidden),
            torch.nn.ReLU()
        )
        
        self.backBone = torch.nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = torch.nn.Sequential(
            torch.nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        self.valueHead = torch.nn.Sequential(
            torch.nn.Conv2d(num_hidden, 3, kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3 * game.row_count * game.column_count, 1),
            torch.nn.Tanh()
        )
        #self.to(device)
    # def policyHeadFunc (self, a, num_hidden: int, game):
    #     # import IPython; IPython.embed()
    #     print(a.size())
    #     a = torch.nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1)(a)
    #     print(a.size())
    #     a = torch.nn.BatchNorm2d(32)(a)
    #     print(a.size())
    #     a =torch.nn.ReLU()(a)
    #     print(a.size())
    #     a =torch.nn.Flatten()(a)
    #     print(a.size())
    #     a =torch.nn.Linear(32 * game.row_count * game.column_count, game.action_size)(a)
    #     print(a.size())
    #     return a
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        # policy = self.policyHeadFunc(x, self.numHIdden, self.ggame)
        value = self.valueHead(x)
        return policy, value
        
class ResBlock(torch.nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_hidden)
        self.conv2 = torch.nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_hidden)
    def forward(self, x):
        #WHY DO We DO RESIDUAL???????? NEED TO KNOW
        residual = x
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.nn.functional.relu(x)
        return x