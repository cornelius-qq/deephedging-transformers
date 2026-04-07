import torch
from payoffs import *


def compute_gains(model, S, K, T, N_steps, payoff_fn, premium=0.0, c=0.001):

    N_paths = S.shape[0]
    dt      = T / N_steps

    # No position held before the first step
    prev_delta = torch.zeros(N_paths)    # [N_paths]

    pnl        = torch.zeros(N_paths)    # accumulated P&L from delta hedging
    trans_cost = torch.zeros(N_paths)    # accumulated transaction costs

    for t in range(N_steps):

        # Normalised time remaining: 1.0 at t=0, decreasing toward 0
        time_left = torch.full((N_paths,), (N_steps - t) * dt / T)

        # Moneyness: current stock price relative to strike
        kappa = S[:, t] / K              # [N_paths]

        # Query the policy: returns delta_t for the full batch
        delta_t = model(kappa, time_left, prev_delta)   # [N_paths]

        # P&L: hold delta_t shares from t to t+1
        pnl += delta_t * (S[:, t+1] - S[:, t])

        # Proportional transaction cost for the rebalance
        # cost = c * |new_position - old_position| * current_price
        trans_cost += c * torch.abs(delta_t - prev_delta) * S[:, t]

        # Store detached delta_t for next step.
        # .detach() cuts the gradient graph across time steps (truncated BPTT):
        # without it, backprop would unroll through every previous step,
        # multiplying memory and time by N_steps.
        prev_delta = delta_t.detach()

    # Terminal payoff owed by the hedger (e.g. call(S, K) for a short call)
    payoff = payoff_fn(S)                # [N_paths]

    # Gain = premium received + trading P&L - transaction costs - payoff owed
    gains = premium + pnl - trans_cost - payoff

    return gains
