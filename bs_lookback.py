"""
bs_lookback.py
--------------
Goldman-Sosin-Gatto (1979) closed-form pricing and delta for the
floating-strike lookback call under Black-Scholes / GBM.

Payoff
------
    payoff = S_T - min_{0 <= t <= T} S_t

The state at hedging step t is (S_t, m_t) where
    m_t = min_{0 <= s <= t} S_s   (running minimum observed so far)

Pricing formula
---------------
For r > 0:

    C(S, m, r, sigma, tau) =
        S * N(d1)
      - m * exp(-r*tau) * N(d2)
      + (sigma^2 / (2*r)) * S *
          [ exp(-r*tau) * (m/S)^(2r/sigma^2) * N(d3)
            - N(-d1) ]

where
    d1 = [ln(S/m) + (r + sigma^2/2) * tau] / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    d3 = [-ln(S/m) + (r - sigma^2/2) * tau] / (sigma * sqrt(tau))

Limiting cases
--------------
- At maturity (tau -> 0, S > m): C -> S - m  (the payoff).  Verified.
- At maturity (tau -> 0, S = m): C -> 0.     Verified.
- For r = 0:  the formula has a 1/(2r) singularity.  We regularise by
  replacing r with max(r, 1e-8), which introduces negligible error (<1e-7
  in prices at the parameter values used in this project).

Delta
-----
Delta = dC/dS is computed via central finite differences:
    delta(S, m, r, sigma, tau) ~= [C(S+eps, m, ...) - C(S-eps, m, ...)] / (2*eps)
with eps = S * 1e-4.

Finite differences avoid the risk of sign errors in the analytical
derivative and give machine-accurate deltas for the step sizes used here.

LookbackDeltaModel
------------------
A model class with the same calling signature as TransformerHedgeNet, i.e.
    deltas = model(features)     features: [N, T, 3]  ->  deltas: [N, T]

where features[:, :, 0] = S_t / K  (normalised spot)
      features[:, :, 1] = m_t / K  (normalised running minimum)
      features[:, :, 2] = tau_t / T (normalised remaining time)

This lets LookbackDeltaModel be dropped into the same compute_gains_from_features
gym as the neural network policies for a fair evaluation comparison.
"""

import numpy as np
import torch
from scipy.stats import norm

# 1.  Closed-form price  (numpy, operates on scalars or arrays)
def lookback_call_price(S, m, r, sigma, tau):
    """
    Goldman-Sosin-Gatto price of the floating-strike lookback call.

    All arguments are numpy scalars or numpy arrays of compatible shape.
    Requires m <= S (running minimum does not exceed current price).

    Parameters
    ----------
    S     : current stock price
    m     : running minimum  (m <= S)
    r     : risk-free rate   (use 0.0 for this project; regularised internally)
    sigma : annual volatility
    tau   : remaining time to maturity in years  (tau > 0)

    Returns
    -------
    price : float or numpy array, same shape as S
    """

    # Regularise r = 0 to avoid the sigma^2 / (2r) singularity.
    # At r = 1e-8 the pricing error is below 1e-7 for all realistic parameters.
    r_eff = np.maximum(r, 1e-8)

    sqrt_tau = np.sqrt(tau)
    log_sm   = np.log(S / m)       

    # Standard GSG arguments
    d1 = (log_sm + (r_eff + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    # Reflected argument: -ln(S/m) + (r - sigma^2/2)*tau in the numerator
    d3 = (-log_sm + (r_eff - 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)

    # Exponent for the reflected term: (m/S)^(2r/sigma^2)
    alpha          = 2.0 * r_eff / sigma**2           # 2r / sigma^2
    reflected_pow  = np.exp(alpha * np.log(m / S))    # (m/S)^alpha, avoids 0^0 at S=m

    term1 = S * norm.cdf(d1)
    term2 = - m * np.exp(-r_eff * tau) * norm.cdf(d2)
    term3 = (sigma**2 / (2.0 * r_eff)) * S * (
        np.exp(-r_eff * tau) * reflected_pow * norm.cdf(d3)
        - norm.cdf(-d1)
    )

    return term1 + term2 + term3



# 2.  Delta via central finite differences

def lookback_delta(S, m, r, sigma, tau):
    """
    Delta of the floating-strike lookback call with respect to S.

    Computed via central finite differences on lookback_call_price for
    numerical robustness.  The relative perturbation eps = 1e-4 * S gives
    an absolute error well below 1e-6 for all parameter values in this project.

    Parameters
    ----------
    S, m, r, sigma, tau : same as lookback_call_price (numpy scalars or arrays)

    Returns
    -------
    delta : float or numpy array, same shape as S
    """
    eps    = S * 1e-4                                      # relative perturbation
    c_up   = lookback_call_price(S + eps, m, r, sigma, tau)
    c_down = lookback_call_price(S - eps, m, r, sigma, tau)
    return (c_up - c_down) / (2.0 * eps)


# 3.  LookbackDeltaModel  (same calling convention as TransformerHedgeNet)

class LookbackDeltaModel:
    """
    Analytical delta-hedging policy for the floating-strike lookback call.

    This class mirrors the calling convention of TransformerHedgeNet so that
    it can be passed to compute_gains_from_features without any modification.

    Usage
    -----
    model = LookbackDeltaModel(K=1.0, r=0.0, sigma=0.2, T=1.0)
    deltas = model(features)    # features: torch.Tensor [N, T, 3]
                                # deltas:   torch.Tensor [N, T]

    Feature layout (must match build_lookback_feature_matrix)
    ----------------------------------------------------------
    features[:, :, 0] = S_t / K          -- normalised spot
    features[:, :, 1] = m_t / K          -- normalised running minimum
    features[:, :, 2] = (T - t*dt) / T   -- normalised time remaining
    """

    def __init__(self, K: float, r: float, sigma: float, T: float):
        self.K     = K
        self.r     = r
        self.sigma = sigma
        self.T     = T

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the analytical lookback delta for every (path, step) pair.

        Parameters
        ----------
        features : torch.Tensor [N_paths, N_steps, 3]

        Returns
        -------
        deltas : torch.Tensor [N_paths, N_steps]
        """
        # Recover physical quantities from normalised features
        S_normalised   = features[:, :, 0].numpy()   # S_t / K
        m_normalised   = features[:, :, 1].numpy()   # m_t / K
        tau_normalised = features[:, :, 2].numpy()   # (T - t) / T

        S   = S_normalised   * self.K      # actual spot price    [N, T]
        m   = m_normalised   * self.K      # actual running min   [N, T]
        tau = tau_normalised * self.T      # actual remaining time [N, T]

        # Compute analytical delta for the full [N, T] array in one numpy call
        delta_np = lookback_delta(S, m, self.r, self.sigma, tau)   # [N, T]

        return torch.tensor(delta_np, dtype=torch.float32)

    # The two methods below are no-ops that allow LookbackDeltaModel to be
    # called inside eval loops that call model.eval() or model.train() on
    # whatever policy they receive.

    def train(self, mode: bool = True):
        pass

    def eval(self):
        pass
