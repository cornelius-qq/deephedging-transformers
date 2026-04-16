"""
compare_lookback.py
-------------------
Three-way comparison of hedging strategies for the floating-strike lookback call:

  1. Analytical delta  -- Goldman-Sosin-Gatto closed form, applied at discrete steps.
                          This is the "discretised perfect hedge": optimal in continuous
                          time but subject to discretisation error at N_steps=50.

  2. MLP (feedforward) -- MLPHedgeNet trained with CVaR minimisation.
                          State: (S_t/K, m_t/K, tau_t).
                          Each timestep processed independently (no temporal memory).

  3. Transformer       -- TransformerHedgeNet trained with CVaR minimisation.
                          Same state features as the MLP, but with causal self-attention
                          so delta_t can depend on the entire price history up to t.

All three strategies are evaluated on the same held-out GBM paths with
proportional transaction costs c=0.001.

Why the lookback is a good benchmark
--------------------------------------
- The payoff is path-dependent via m_T = min_{0<=t<=T} S_t.
- The analytical delta at step t depends on both S_t and m_t = running minimum.
- A memoryless model (MLP) already has access to m_t as an explicit feature, so
  it can in principle learn the correct delta without memory.
- The Transformer has causal attention over the entire price sequence, so it can
  additionally exploit correlations across time that the MLP ignores.
- The attention heatmap for the Transformer should reveal which past price levels
  the model attends to -- we expect strong attention to the step achieving the minimum.

Output
------
- Overlaid gain distribution plot  (saved to compare_lookback_gains.png)
- Training convergence plot for both neural models  (saved to compare_lookback_training.png)
- Summary table printed to stdout: CVaR, mean gain, std gain for all three strategies
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from simulate            import simulate_gbm
from payoffs             import lookback_call
from bs_lookback         import lookback_call_price, LookbackDeltaModel
from gym_transformer     import (build_lookback_feature_matrix,
                                  compute_gains_from_features)
from loss                import cvar
from network             import MLPHedgeNet
from network_transformer import TransformerHedgeNet


# Parameters

S0         = 1.0
K          = 1.0    # used only for normalisation; not a strike in the payoff
r          = 0.0
sigma      = 0.2
T          = 1.0
N_steps    = 50

N_train    = 20_000
N_val      = 5_000
batch_size = 5_000
epochs     = 2_000
lr         = 1e-3
alpha      = 0.05   # CVaR tail probability
c          = 0.001  # proportional transaction cost



# Lookback premium at time 0

# At inception the running minimum equals S0, so we evaluate at (S, m) = (S0, S0).

lookback_premium = float(lookback_call_price(S0, S0, r, sigma, T))
print(f"Lookback call premium (GSG): {lookback_premium:.4f}")
print(f"  (for reference: ATM BS call premium = "
      f"{float(lookback_call_price(S0, S0, max(r, 1e-8), sigma, T)):.4f})")



# Simulate training and evaluation paths  (fixed seed for reproducibility)

torch.manual_seed(42)
S_train = simulate_gbm(N_train, S0, r, sigma, T, N_steps)   # [N_train, N_steps+1]

torch.manual_seed(99)
S_eval  = simulate_gbm(N_val,   S0, r, sigma, T, N_steps)   # [N_val,   N_steps+1]

# Build the feature matrix for the evaluation paths once; reused by all three models.
features_eval = build_lookback_feature_matrix(S_eval, K, T, N_steps)  # [N_val, N_steps, 3]


# Helper: compute empirical CVaR at level alpha from a gains tensor

def empirical_cvar(gains: torch.Tensor, alpha: float = 0.05) -> float:
    """
    Expected gain in the worst alpha-fraction of outcomes.
    A more negative value means worse tail performance.
    """
    sorted_gains = torch.sort(gains).values
    n_tail       = max(1, int(alpha * len(gains)))
    return float(sorted_gains[:n_tail].mean())



# Shared training loop  (used for both MLP and Transformer)

def train_model(model, model_name: str) -> tuple:
    """
    Train model on the lookback call with CVaR loss.

    Both MLPHedgeNet and TransformerHedgeNet share this loop because they
    have the same calling convention: model(features) -> [N, T].

    Parameters
    ----------
    model       : MLPHedgeNet or TransformerHedgeNet
    model_name  : str  -- used for progress printing

    Returns
    -------
    eta     : nn.Parameter  -- optimised VaR level
    history : dict          -- {'epoch': [...], 'loss': [...]}
    """
    # eta is the VaR level, jointly optimised with model weights (Rockafellar-Uryasev)
    eta       = nn.Parameter(torch.tensor(0.0))
    optimiser = torch.optim.Adam(
        list(model.parameters()) + [eta],
        lr = lr,
    )

    history = {'epoch': [], 'loss': []}

    for epoch in range(epochs):
        model.train()

        # Random mini-batch from the pre-simulated training pool
        batch_idx = torch.randint(0, N_train, (batch_size,))
        S_batch   = S_train[batch_idx]                         # [batch_size, N_steps+1]

        # Build lookback feature matrix for this mini-batch
        features_batch = build_lookback_feature_matrix(S_batch, K, T, N_steps)

        # Single forward pass: model(features) -> [batch_size, N_steps]
        gains_batch = compute_gains_from_features(
            model       = model,
            features    = features_batch,
            S           = S_batch,
            payoff_fn   = lookback_call,
            premium     = lookback_premium,
            c           = c,
        )

        loss = cvar(gains_batch, eta, alpha)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        history['epoch'].append(epoch)
        history['loss'].append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"  [{model_name}] Epoch {epoch+1:4d}/{epochs}"
                  f"  |  CVaR loss: {loss.item():+.4f}"
                  f"  |  eta: {eta.item():+.4f}")

    return eta, history


# 1.  Analytical delta benchmark  (no training required)

print("\n--- Analytical delta benchmark ---")
analytical_model = LookbackDeltaModel(K=K, r=r, sigma=sigma, T=T)

with torch.no_grad():
    gains_analytical = compute_gains_from_features(
        model       = analytical_model,
        features    = features_eval,
        S           = S_eval,
        payoff_fn   = lookback_call,
        premium     = lookback_premium,
        c           = c,
    )
print(f"  CVaR:  {empirical_cvar(gains_analytical, alpha):+.4f}")
print(f"  Mean:  {gains_analytical.mean().item():+.4f}")
print(f"  Std:   {gains_analytical.std().item():.4f}")



# 2.  MLP training

print("\n--- Training MLP ---")

# n_features=3: (S_t/K, m_t/K, tau_t) -- same 3 features as the Transformer.
# hidden_size=64: matches the Transformer's d_model for a fair parameter budget.
mlp_model = MLPHedgeNet(n_features=3, hidden_size=64)

n_params_mlp = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
print(f"  MLPHedgeNet | trainable parameters: {n_params_mlp:,}")

eta_mlp, history_mlp = train_model(mlp_model, "MLP")

# Evaluate on held-out paths
mlp_model.eval()
with torch.no_grad():
    gains_mlp = compute_gains_from_features(
        model       = mlp_model,
        features    = features_eval,
        S           = S_eval,
        payoff_fn   = lookback_call,
        premium     = lookback_premium,
        c           = c,
    )
print(f"  CVaR:  {empirical_cvar(gains_mlp, alpha):+.4f}")
print(f"  Mean:  {gains_mlp.mean().item():+.4f}")
print(f"  Std:   {gains_mlp.std().item():.4f}")


# 3.  Transformer training

print("\n--- Training Transformer ---")

# n_features=3 matches build_lookback_feature_matrix: (S/K, m/K, tau).
# Architecture hyperparameters match train_transformer.py baseline.
transformer_model = TransformerHedgeNet(
    n_features = 3,
    d_model    = 64,
    n_heads    = 4,
    d_ff       = 256,
    n_blocks   = 2,
    max_len    = N_steps + 1,
)

n_params_tf = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
print(f"  TransformerHedgeNet | trainable parameters: {n_params_tf:,}")

eta_tf, history_tf = train_model(transformer_model, "Transformer")

# Evaluate on held-out paths
transformer_model.eval()
with torch.no_grad():
    gains_transformer = compute_gains_from_features(
        model       = transformer_model,
        features    = features_eval,
        S           = S_eval,
        payoff_fn   = lookback_call,
        premium     = lookback_premium,
        c           = c,
    )
print(f"  CVaR:  {empirical_cvar(gains_transformer, alpha):+.4f}")
print(f"  Mean:  {gains_transformer.mean().item():+.4f}")
print(f"  Std:   {gains_transformer.std().item():.4f}")



# Summary table

print("\n" + "=" * 60)
print(f"{'Strategy':<22} {'CVaR':>10} {'Mean':>10} {'Std':>10}")
print("-" * 60)
for name, gains in [
    ("Analytical delta", gains_analytical),
    ("MLP",              gains_mlp),
    ("Transformer",      gains_transformer),
]:
    print(f"{name:<22} "
          f"{empirical_cvar(gains, alpha):>10.4f} "
          f"{gains.mean().item():>10.4f} "
          f"{gains.std().item():>10.4f}")
print("=" * 60)


# Plot 1:  Gain distributions (overlaid histograms)

fig, ax = plt.subplots(figsize=(11, 5))

all_gains = np.concatenate([
    gains_analytical.numpy(),
    gains_mlp.numpy(),
    gains_transformer.numpy(),
])
bins = np.linspace(all_gains.min(), all_gains.max(), 80)

ax.hist(gains_analytical.numpy(),  bins=bins, alpha=0.45,
        label='Analytical delta (GSG)', color='darkorange')
ax.hist(gains_mlp.numpy(),         bins=bins, alpha=0.45,
        label='MLP (trained)',          color='seagreen')
ax.hist(gains_transformer.numpy(), bins=bins, alpha=0.45,
        label='Transformer (trained)',  color='steelblue')

# Mark the CVaR for each strategy with a vertical line
ax.axvline(empirical_cvar(gains_analytical, alpha), color='darkorange',
           linestyle='--', linewidth=1.2,
           label=f'CVaR (analytical) = {empirical_cvar(gains_analytical, alpha):.3f}')
ax.axvline(empirical_cvar(gains_mlp, alpha), color='seagreen',
           linestyle='--', linewidth=1.2,
           label=f'CVaR (MLP)         = {empirical_cvar(gains_mlp, alpha):.3f}')
ax.axvline(empirical_cvar(gains_transformer, alpha), color='steelblue',
           linestyle='--', linewidth=1.2,
           label=f'CVaR (Transformer) = {empirical_cvar(gains_transformer, alpha):.3f}')

ax.set_xlabel('Gain  Z = premium + PnL - costs - payoff', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(
    f'Lookback call: gain distributions  '
    f'(N_val={N_val}, c={c}, alpha={alpha}, N_steps={N_steps})',
    fontsize=12,
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('compare_lookback_gains.png', dpi=150)
print("\nSaved: compare_lookback_gains.png")
plt.show()



# Plot 2:  Training convergence for MLP and Transformer

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(history_mlp['epoch'],     history_mlp['loss'],
        linewidth=0.9, color='seagreen',  label='MLP')
ax.plot(history_tf['epoch'],      history_tf['loss'],
        linewidth=0.9, color='steelblue', label='Transformer')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('CVaR loss', fontsize=12)
ax.set_title('Training convergence: MLP vs Transformer (lookback call)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linewidth=0.4, alpha=0.6)
plt.tight_layout()
plt.savefig('compare_lookback_training.png', dpi=150)
print("Saved: compare_lookback_training.png")
plt.show()
