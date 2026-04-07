import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


S0      = 1.0    # initial stock price (normalised to 1)
K       = 1.0    # strike price        (ATM since S0 = K)
r       = 0.0    # risk-free rate      (zero keeps things clean)
sigma   = 0.2    # annual volatility   (20%)
T       = 1.0    # option maturity     (1 year)
N_steps = 50     # hedging steps       (roughly weekly)
N_train = 20000  # paths for training
N_val   = 5000   # paths for validation


def simulate_gbm(N_paths,S0, r, sigma, T, N_steps):

    dt = T / N_steps

    # One standard normal draw per path per step
    Z = torch.randn(N_paths, N_steps)

    # Log-returns: each column is one time increment
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum gives log(S_t / S_0); exponentiate to get S_t
    log_S     = torch.cumsum(log_increments, dim=1)
    S_future  = S0 * torch.exp(log_S)

    # Prepend S_0 column so index 0 = time 0, index N_steps = maturity
    S0_col = torch.full((N_paths, 1), S0)


    return torch.cat([S0_col, S_future], dim=1)
