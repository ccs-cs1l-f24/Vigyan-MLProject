import numpy
import math
import Cycles
import MCTS
import torch
import torch.nn
import torch.nn.functional

class ResNetCycles(torch.nn.Module):
    def __init__(self, game: Cycles.Cycles, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.row_count = game.row_count
        self.col_count = game.column_count
        self.startBlock = torch.nn.Sequential(
            RowColumnConv(3, num_hidden, kernal_row=self.row_count, kernal_col=self.col_count),
            torch.nn.BatchNorm2d(num_hidden), #maybe choose different norm function for this
            torch.nn.ReLU()
        )
        
        self.backBone = torch.nn.ModuleList(
            [ResBlock(num_hidden, row_count=self.row_count, col_count=self.col_count) for i in range(num_resBlocks)]
        )
        self.policyHead = torch.nn.Sequential(
            RowColumnConv(num_hidden, 32, kernal_row=self.row_count, kernal_col=self.col_count),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        self.valueHead = torch.nn.Sequential(
            RowColumnConv(num_hidden, 3, kernal_row=self.row_count, kernal_col=self.col_count),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3 * game.row_count * game.column_count, 1),
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
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        # policy = self.policyHeadFunc(x, self.numHIdden, self.ggame)
        value = self.valueHead(x)
        return policy, value
    def __repr__(self):
        return "ResNetCycles"
        
class ResBlock(torch.nn.Module):
    def __init__(self, num_hidden, row_count, col_count):
        super().__init__()
        self.row_count = row_count
        self.col_count = col_count
        self.conv1 = RowColumnConv(num_hidden, num_hidden, kernal_row=self.row_count, kernal_col=self.col_count)
        self.bn1 = torch.nn.BatchNorm2d(num_hidden)
        self.conv2 = RowColumnConv(num_hidden, num_hidden, kernal_row=self.row_count, kernal_col=self.col_count)
        self.bn2 = torch.nn.BatchNorm2d(num_hidden)
    def forward(self, x):
        #The residual is the whole point of the res-net, it helps with normalization
        residual = x
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.nn.functional.relu(x)
        return x
    
class RowColumnConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernal_row, kernal_col):
        super().__init__()
        self.kernal_row = kernal_row
        self.kernal_col = kernal_col
        self.row_conv = torch.nn.Conv2d(in_channels, out_channels, (kernal_row,1))
        self.col_conv = torch.nn.Conv2d(in_channels, out_channels, (1,kernal_col))
    def forward(self, x):
        row_output = self.row_conv(x)
        row_output = row_output.repeat(1,1,self.kernal_row,1)
        col_output = self.col_conv(x)
        col_output = col_output.repeat(1,1,1,self.kernal_col)
        # print("row\n",row_output)
        # print("col\n",col_output)
        return (row_output+col_output)