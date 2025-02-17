import torch
import numpy
import LineCycleMaker
import IdealLinePlayer

# wad = numpy.array([1,2,3,2,1,1,1,1,1])
# player = IdealLinePlayer.IdealLinePlayer()
# player.action(wad)
# wadawd = [[],[]]
# wadawd[0].append('r')
valid_moves = [1,2]
print(numpy.sum(valid_moves))
# print('ww',wadawd)


# print(LineCycleMaker.CircleGraph(5))
# print(LineCycleMaker.LineGraph(5))
# print("\n")
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")

# import numpy
# print(numpy.__version__)

# print(torch.__version__)

# import tqdm
# print(tqdm.__version__)

# import torch
# import torch.nn as nn
# import torch.nn.functional

# ad = torch.randn(2, 3, 5, 7, 11, 3,3,3,3,3,3,3,3,3)
# m = nn.Flatten()
# # With default parameters
# output = m(ad)
# print(output.size())