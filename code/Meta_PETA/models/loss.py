import torch
import torch.nn as nn

def SquareError(scores, targets):
    SE = torch.square(scores - targets)
    return SE

def AbsolError(scores, targets):
    AE = torch.abs(scores - targets)

    return AE

def AbsolPercentageError(scores, targets):
    targets = torch.abs(targets)
    AE = torch.abs(scores - targets)
    PAE = AE / targets

    return PAE

