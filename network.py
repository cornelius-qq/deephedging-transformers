import torch
import torch.nn as nn

class DeltaHedgeNet(nn.Module):

    def __init__(self):
        super().__init__()          # always required — initialises nn.Module

        self.net = nn.Sequential(   # stack layers in order
            nn.Linear(3, 64),       # 3 inputs (moneyness, time_left, prev_delta) → 64 neurons
            nn.Tanh(),              # activation
            nn.Linear(64, 64),      # 64 → 64
            nn.Tanh(),              # activation
            nn.Linear(64, 1),       # 64 → 1 output (the delta)
        )

    def forward(self, moneyness, time_left, prev_delta):

        # moneyness, time_left, prev_delta are all shape [N]
        # combine them into one tensor of shape [N, 3]
        x = torch.stack([moneyness, time_left, prev_delta], dim=1)  # [N, 3]

        # pass through all layers
        out = self.net(x)           # [N, 1]

        # remove the last dimension so shape is [N] not [N, 1]
        return out.squeeze(1)
