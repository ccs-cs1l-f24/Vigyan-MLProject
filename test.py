import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# import torch
# import torch.nn as nn
# import torch.nn.functional

# ad = torch.randn(2, 3, 5, 7, 11, 3,3,3,3,3,3,3,3,3)
# m = nn.Flatten()
# # With default parameters
# output = m(ad)
# print(output.size())