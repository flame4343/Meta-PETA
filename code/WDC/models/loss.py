import torch
import torch.nn as nn


class MAPE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None):
        ape = (target - pred).abs() / (target.abs() + self.eps)
        if weight is None:
            return 100 * torch.mean(ape)
        else:
            return (ape * weight / (weight.sum())).sum()
class SMAPE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None):
        sape = 2 * (target - pred).abs() / (pred.abs() + target.abs() + self.eps)
        if weight is None:
            return 100 * torch.mean(sape)
        else:
            return (sape * weight / (weight.sum())).sum()
class MY_MAPE_FT(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None):
        sape = (target - pred).abs() / (pred.abs() + self.eps)
        if weight is None:
            return 100 * torch.mean(sape)
        else:
            return (sape * weight / (weight.sum())).sum()
class SatsfScore(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None):
        sape = 2 * (target - pred).abs() / (pred.abs() + target.abs() + self.eps)
        sape = torch.where(torch.gt(target, pred), 2 * sape, 1 * sape)
        if weight is None:
            return 100 * torch.mean(sape)
        else:
            return (sape * weight / (weight.sum())).sum()
