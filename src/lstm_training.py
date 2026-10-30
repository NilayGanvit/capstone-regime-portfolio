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
from pathlib import Path

import pandas as pd

from features import build_feature_matrix
from regime import GaussianHMM
from asset_regime import fit_asset_hmms, high_variance_probabilities
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
    asset_regime_states: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Train {"baseline": model, "regime": model} on returns[:initial_window]
    and return them with the full-history feature matrix, ready for
    run_walk_forward's lstm_models/lstm_feature_matrix.

    `asset_regime_states`, if set, also trains "asset_regime" (trial 20): the
    regime recipe with each asset's high-variance probability from its own
    HMM (asset_regime.py) in place of the joint state probabilities. It is
    trained after the other two, each from its own seeded init, so they are
    unchanged."""
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
    variants = [("baseline", train_features), ("regime", regime_features_train)]
    if asset_regime_states is not None:
        asset_hmms = fit_asset_hmms(train_returns.to_numpy(), asset_regime_states)
        asset_prob_df = pd.DataFrame(
            high_variance_probabilities(asset_hmms, train_returns.to_numpy()), index=train_returns.index,
            columns=[f"high_variance_prob_{t}" for t in returns.columns],
        )
        variants.append(("asset_regime", train_features.join(asset_prob_df, how="inner")))

    models = {}
    for variant_name, feats in variants:
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


def save_checkpoint(model: RiskBudgetLSTM, path: Path, *, seed: int, variant: str, epochs: int) -> None:
    """Save a trained allocator in the same layout as the archived frozen-reference
    checkpoints (state dict plus the settings needed to rebuild it), so a run can be
    replayed without retraining."""
    import torch

    torch.save({
        "seed": seed, "variant": variant, "budget_floor": model.budget_floor, "epochs": epochs,
        "n_features": model.lstm.input_size, "n_assets": model.n_assets, "hidden_size": model.lstm.hidden_size,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: Path) -> RiskBudgetLSTM:
    """Rebuild an allocator from save_checkpoint (or an archived checkpoint) for inference.
    Loaded with weights_only=True, so only tensors and plain values are read."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = RiskBudgetLSTM(n_features=ck["n_features"], n_assets=ck["n_assets"], hidden_size=ck.get("hidden_size", 32),
                           random_state=None, budget_floor=ck["budget_floor"])
    model.load_state_dict(ck["state_dict"])
    return model.double().eval()
