import numpy
import math
import Cycles
import MCTS
import torch
import torch.nn
import torch.nn.functional

class ResNetLinear(torch.nn.Module):
    # numHIdden = 1
    def __init__(self, game: Cycles.Cycles, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        # self.numHIdden = num_hidden
        #self.ggame = game
        self.startBlock = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*game.row_count*game.column_count, num_hidden),
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU()
        )
        
        self.backBone = torch.nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, game.action_size)
        )
        self.valueHead = torch.nn.Sequential(
            torch.nn.Linear(num_hidden,3),
            torch.nn.BatchNorm1d(3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
            torch.nn.Tanh()
        )
        self.to(device)
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
        # print("size of x: ",x.size())
        for resBlock in self.backBone:
            x = resBlock(x)
        # print("size of x after: ",x.size())
        policy = self.policyHead(x)
        # policy = self.policyHeadFunc(x, self.numHIdden, self.ggame)
        value = self.valueHead(x)
        return policy, value
    def __repr__(self):
        return "ResNetLinear"
        
class ResBlock(torch.nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = torch.nn.Linear(num_hidden, num_hidden)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)
        self.conv2 = torch.nn.Linear(num_hidden, num_hidden)
        self.bn2 = torch.nn.BatchNorm1d(num_hidden)
    def forward(self, x):
        #The residual is the whole point of the res-net, it helps with normalization
        residual = x
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.nn.functional.relu(x)
        return x