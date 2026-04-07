"""
gym_transformer.py
------------------
Hedging gym for the Causal Transformer policy.

The key difference from gym.py (MLP gym) is that the Transformer processes
the FULL sequence of features in a single forward pass, returning all T
hedge ratios at once. This is possible because the causal mask inside the
attention layers ensures that delta_t depends only on features at times
0, ..., t -- there is no lookahead.

Feature vector at each time step:
    x_t = (kappa_t, tau_t)  in R^2
where:
    kappa_t = S_t / K           moneyness
    tau_t   = (T - t*dt) / T   normalised time remaining

We use n_features=2 (dropping prev_delta) so that the full feature matrix
[N, T, 2] can be built from the path tensor S in one vectorised operation
before the model is called. This avoids the circular dependency that arises
when prev_delta is included: delta_{t-1} is a model output, so you cannot
build the full input matrix without already having run the model.

The Transformer's causal attention provides implicit memory of the full path
history at each position, making prev_delta redundant: the model can infer
its own previous decisions from the sequence of (kappa, tau) features.

Transaction costs are still computed in a loop over time steps using the
returned delta sequence, since the cost at step t depends on |delta_t - delta_{t-1}|.
"""

import torch
from typing import Callable


def build_feature_matrix(S: torch.Tensor,
                         K: float,
                         T: float,
                         N_steps: int) -> torch.Tensor:
    """
    Build the full feature matrix [N, T, 2] from the path tensor S.

    At each time step t the feature vector is (kappa_t, tau_t):
        kappa_t = S_t / K          -- moneyness
        tau_t   = (T - t*dt) / T  -- normalised time to maturity

    Parameters
    ----------
    S        : torch.Tensor [N_paths, N_steps + 1]
    K        : float  -- strike price
    T        : float  -- time to maturity in years
    N_steps  : int    -- number of hedging steps

    Returns
    -------
    features : torch.Tensor [N_paths, N_steps, 2]
    """
    N_paths = S.shape[0]
    dt = T / N_steps

    # Moneyness at each step t = 0, ..., N_steps-1: shape [N_paths, N_steps]
    moneyness = S[:, :N_steps] / K  # kappa_t = S_t / K

    # Normalised time remaining at each step: shape [N_steps]
    # At t=0: tau = N_steps * dt / T = 1.0
    # At t=N_steps-1: tau = 1 * dt / T = dt/T (near zero)
    time_steps  = torch.arange(N_steps, dtype=torch.float32)           # [N_steps]
    tau         = ((N_steps - time_steps) * dt / T)                    # [N_steps]
    tau         = tau.unsqueeze(0).expand(N_paths, -1)                 # [N_paths, N_steps]

    # Stack along the feature dimension: [N_paths, N_steps, 2]
    features = torch.stack([moneyness, tau], dim=2)

    return features


def compute_gains_transformer(model,
                              S: torch.Tensor,
                              K: float,
                              T: float,
                              N_steps: int,
                              payoff_fn: Callable[[torch.Tensor], torch.Tensor],
                              premium: float,
                              c: float = 0.001) -> torch.Tensor:
    """
    Roll out the Transformer hedging strategy and compute realised gains.

    The model is called ONCE with the full feature sequence [N, T, 2] and
    returns all hedge ratios [N, T] simultaneously. The gains and transaction
    costs are then computed from the returned delta sequence.

    Parameters
    ----------
    model : TransformerHedgeNet
        Causal Transformer policy. Called as model(features) where
        features has shape [N_paths, N_steps, 2].
        Returns deltas of shape [N_paths, N_steps].
    S : torch.Tensor [N_paths, N_steps + 1]
        Simulated stock price paths.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    N_steps : int
        Number of hedging steps.
    payoff_fn : Callable
        Function of S returning the terminal payoff [N_paths].
    premium : float
        Option premium received at time 0.
    c : float
        Proportional transaction cost coefficient. Default 0.001.

    Returns
    -------
    gains : torch.Tensor [N_paths]
        Realised gain for each path.
    """
    N_paths = S.shape[0]

    # Build the full feature matrix in one vectorised operation: [N_paths, N_steps, 2]
    features = build_feature_matrix(S, K, T, N_steps)

    # Single forward pass: Transformer outputs all T hedge ratios at once
    # deltas[n, t] depends only on features[n, 0:t+1] due to the causal mask
    deltas = model(features)  # [N_paths, N_steps]

    # Compute P&L from the hedge: sum over time of delta_t * (S_{t+1} - S_t)
    # dS has shape [N_paths, N_steps]: price changes at each step
    dS  = S[:, 1:] - S[:, :-1]        # [N_paths, N_steps]
    pnl = torch.sum(deltas * dS, dim=1)  # [N_paths]

    # Compute transaction costs: c * |delta_t - delta_{t-1}| * S_t
    # delta_{-1} = 0 (no position before the first step)
    prev_deltas = torch.cat([
        torch.zeros(N_paths, 1),   # delta_{-1} = 0 for all paths
        deltas[:, :-1]             # delta_0, ..., delta_{T-2}
    ], dim=1)                      # [N_paths, N_steps]

    trans_cost = c * torch.sum(
        torch.abs(deltas - prev_deltas) * S[:, :N_steps],
        dim=1
    )  # [N_paths]

    # Terminal payoff owed by the hedger
    payoff = payoff_fn(S)  # [N_paths]

    # Full gain formula
    gains = premium + pnl - trans_cost - payoff  # [N_paths]

    return gains
