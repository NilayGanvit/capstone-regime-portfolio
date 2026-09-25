"""
Run the LSTM risk-budgeting allocator's end-to-end training loop on the
REAL ten-ETF universe.

Companion to scripts/run_real_data.py (ERC) and scripts/run_robustness_checks.py.
Closes a specific gap identified in a paper-vs-code synthesis check against
the M3 draft: allocation_lstm.train_lstm_allocator worked (per
tests/test_allocation_lstm.py) but had never actually been run on the real
universe -- only on synthetic/toy data inside tests. This script does that,
for both the baseline (market features only) and regime (features +
one-step-ahead HMM state probabilities) variants, and reports the training
loss curve as evidence the loop runs and converges on real data.

What this does NOT do: wire the trained model into run_walk_forward as
lstm_baseline/lstm_regime/lstm_blend configs. That remains explicitly
out of scope (see README "Next steps") -- this script's only claim is
"the training loop has now been exercised on the real universe," not
"the LSTM allocator is backtested."

Chronological discipline: per M2/M3 ("these choices will be made before
the final out-of-sample test"), training uses only the initial-training
and validation periods (through data.M2Calendar.validation_end,
2018-12-31); the 2019-2026-08 final-test window is never touched here.
The regime-aware variant's HMM is fit once on the initial-training window
only (matching run_real_data.py's own no-scheduled-refit ERC pass) and
its one-step-ahead predicted probabilities are computed causally across
the full dev+validation slice via GaussianHMM.predicted_probabilities,
which is itself already a forward-only (no smoothing) recursion.

Usage:
    python scripts/run_lstm_training.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, M2Calendar, initial_window_length  # noqa: E402
from features import build_feature_matrix  # noqa: E402
from regime import GaussianHMM  # noqa: E402
from allocation_lstm import (  # noqa: E402
    RiskBudgetLSTM, build_lstm_training_windows, train_lstm_allocator,
    require_torch, require_cvxpylayers,
)

SEQ_LEN = 60
COV_WINDOW = 252
N_EPOCHS = 200


def main() -> None:
    require_torch()
    require_cvxpylayers()

    print("=" * 78)
    print("LSTM TRAINING -- real ten-ETF universe, development+validation only")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    full_returns = ds.returns

    calendar = M2Calendar()
    initial_window = initial_window_length(full_returns.index, calendar)

    # Chronological cutoff: never let the LSTM see the final-test window.
    dev_returns = full_returns.loc[:calendar.validation_end]
    print(f"\nTraining/validation slice: {dev_returns.index.min().date()} to "
          f"{dev_returns.index.max().date()} ({len(dev_returns)} trading days), "
          f"excludes the {calendar.validation_end.date()}-to-{calendar.final_test_end.date()} final test entirely.")

    # HMM fit once on the initial-training window only (no scheduled refit
    # here -- matches run_real_data.py's own no-refit ERC pass; this
    # script's purpose is exercising LSTM training on real data, not
    # re-deriving the regime pipeline's refit schedule).
    train_X = dev_returns.to_numpy()[:initial_window]
    hmm = GaussianHMM(n_states=3, random_state=0).fit(train_X)
    # P(S_t+1 = k | F_t) for every t in dev_returns, causal by construction
    # (predicted_probabilities is filtered_probabilities propagated one
    # step through transmat_, and filtered_probabilities is a forward-only
    # recursion -- see regime.py's own docstrings).
    regime_probs = hmm.predicted_probabilities(dev_returns.to_numpy())
    regime_prob_df = pd.DataFrame(
        regime_probs, index=dev_returns.index,
        columns=[f"regime_prob_{k}" for k in range(3)],
    )

    feature_matrix = build_feature_matrix(dev_returns)
    regime_feature_matrix = feature_matrix.join(regime_prob_df, how="inner")
    print(f"Baseline feature matrix: {feature_matrix.shape}, "
          f"regime-aware feature matrix: {regime_feature_matrix.shape}")

    variants = {
        "baseline": feature_matrix,
        "regime": regime_feature_matrix,
    }

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)

    for variant_name, features in variants.items():
        print(f"\n--- {variant_name} variant ---")
        feature_windows, covariances, fwd_gross_returns, window_dates = build_lstm_training_windows(
            features, dev_returns, seq_len=SEQ_LEN, cov_window=COV_WINDOW,
        )
        print(f"{len(feature_windows)} monthly training windows, "
              f"{pd.Timestamp(window_dates[0]).date()} to {pd.Timestamp(window_dates[-1]).date()}")
        if len(feature_windows) < 12:
            raise RuntimeError(
                f"Only {len(feature_windows)} training windows for the {variant_name} "
                "variant -- too few to draw any conclusion from; check seq_len/cov_window "
                "against the dev+validation slice length."
            )

        n_features = feature_windows[0].shape[1]
        n_assets = fwd_gross_returns[0].shape[0]
        model = RiskBudgetLSTM(n_features=n_features, n_assets=n_assets)

        t0 = time.time()
        result = train_lstm_allocator(
            model, feature_windows, covariances, fwd_gross_returns,
            n_epochs=N_EPOCHS, turnover_penalty=0.1, shrinkage=0.10,
        )
        runtime = time.time() - t0
        loss_history = result["loss_history"]
        print(f"runtime: {runtime:.1f}s over {N_EPOCHS} epochs")
        print(f"loss: epoch 0 = {loss_history[0]:.4f}, "
              f"epoch {N_EPOCHS // 2} = {loss_history[N_EPOCHS // 2]:.4f}, "
              f"epoch {N_EPOCHS - 1} = {loss_history[-1]:.4f}")
        if not np.isfinite(loss_history).all():
            raise RuntimeError(f"Non-finite loss encountered training the {variant_name} variant.")

        pd.Series(loss_history, name="loss").to_csv(
            out_dir / f"lstm_training_loss_{variant_name}.csv", index_label="epoch"
        )

    print(f"\nSaved per-variant loss curves to {out_dir}/lstm_training_loss_<variant>.csv")
    print("\nThis confirms the end-to-end training loop (RiskBudgetLSTM -> "
          "differentiable_risk_budget_layer -> sharpe_turnover_loss) runs and "
          "converges on the real universe, restricted to development+validation "
          "data. The trained model is not wired into run_walk_forward -- see "
          "README 'Next steps'.")


if __name__ == "__main__":
    main()
