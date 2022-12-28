import torch

###################################################
# Loss functions
###################################################
# The loss functions are assumed to take in a matrix where the elements are distances and hence > 0.
# It is also assumed 

def div_2(M):
    # Assumes that M2 > 0
    return 1/(M)+M-2

def logexp_2(M2):
    # Assumes that M2 > 0
    return torch.log(torch.exp(1/M2)+M2)

def div_lin(M):
    # Assumes that M > 0
    return 1/(M)+M-2

def div_log(M):
    # Assumes that M > 0
    return 1/(M)+torch.log(M)-1



