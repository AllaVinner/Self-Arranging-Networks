import torch


def div_2(M2):
    # Assumes that M2 > 0
    return 1/(M2)+M2-2

def logexp_2(M2):
    # Assumes that M2 > 0
    return torch.log(torch.exp(1/M2)+M2)


