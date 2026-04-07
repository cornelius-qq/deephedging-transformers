import torch


def call (S,K):
    return torch.clamp(S[:, -1] - K, min=0.0)


def put (S,K):
    return torch.clamp(K - S[:, -1], min=0.0)


