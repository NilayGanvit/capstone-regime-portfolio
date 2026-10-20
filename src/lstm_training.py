"""
Shared construction of the baseline and regime-aware LSTM allocators.

scripts/run_real_data.py and the seed-robustness script need the same training
recipe (initial window only, HMM regime probabilities as extra features for
the regime variant, Sharpe-minus-turnover loss). Keeping one copy here means a
change to that recipe cannot silently apply to one script and not the other.

Training uses returns[:initial_window] only, never validation or final test --
see scripts/run_real_data.py's module docstring for why.
"""

from __future__ import annotations

import time

import pandas as pd

from features import build_feature_matrix
from regime import GaussianHMM
from allocation_lstm import RiskBudgetLSTM, build_lstm_training_windows, train_lstm_allocator


def train_lstm_variants(
    returns: pd.DataFrame,
    initial_window: int,
    *,
    seed: int = 123,
    budget_floor: float = 0.0,
    n_epochs: int = 200,
    seq_len: int = 60,
    cov_window: int = 252,
    n_states: int = 3,
    turnover_penalty: float = 0.1,
    shrinkage: float = 0.10,
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Train {"baseline": model, "regime": model} on returns[:initial_window]
    and return them with the full-history feature matrix, ready for
    run_walk_forward's lstm_models/lstm_feature_matrix."""
    n_assets = returns.shape[1]
    train_returns = returns.iloc[:initial_window]
    feature_matrix = build_feature_matrix(returns)  # causal by construction, safe over full history
    train_features = feature_matrix.loc[:train_returns.index[-1]]

    hmm = GaussianHMM(n_states=n_states, random_state=0).fit(train_returns.to_numpy())
    regime_probs = hmm.predicted_probabilities(train_returns.to_numpy())
    regime_prob_df = pd.DataFrame(
        regime_probs, index=train_returns.index,
        columns=[f"regime_prob_{k}" for k in range(n_states)],
    )
    regime_features_train = train_features.join(regime_prob_df, how="inner")

    models = {}
    for variant_name, feats in [("baseline", train_features), ("regime", regime_features_train)]:
        windows, covs, fwd, dates = build_lstm_training_windows(
            feats, train_returns, seq_len=seq_len, cov_window=cov_window,
        )
        if verbose:
            print(f"LSTM {variant_name}: {len(windows)} training windows, "
                  f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}")
        model = RiskBudgetLSTM(
            n_features=windows[0].shape[1], n_assets=n_assets, random_state=seed, budget_floor=budget_floor,
        )
        t0 = time.time()
        result = train_lstm_allocator(
            model, windows, covs, fwd, n_epochs=n_epochs,
            turnover_penalty=turnover_penalty, shrinkage=shrinkage,
        )
        if verbose:
            print(f"LSTM {variant_name}: trained in {time.time() - t0:.1f}s, "
                  f"loss {result['loss_history'][0]:.4f} -> {result['loss_history'][-1]:.4f}")
        models[variant_name] = result["model"]
    return models, feature_matrix
