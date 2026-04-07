import torch

def cvar(gains, eta, alpha):
    losses = -gains
    return eta + torch.mean(torch.clamp(losses - eta, min=0.0)) / alpha
