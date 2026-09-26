"""
Validate quarterly LSTM retraining (run_walk_forward's
lstm_retrain_every_n_rebalances) on the real ten-ETF universe, restricted
to the 2015-2018 validation period -- fast enough to run as a sanity
check, unlike a full backtest through the 2026-08-31 final test (~47
quarterly retrains over the full walk, each retraining two LSTM variants
on a growing expanding window).

Compares two arms, same initial-window-trained models as the starting
point for both:
    frozen     : lstm_retrain_every_n_rebalances=None (today's default --
                 train once through 2014, then pure inference)
    quarterly  : lstm_retrain_every_n_rebalances=3, per M2/M3's proposed
                 "retrain the LSTM after the end of the quarter"

This is a capability/correctness check, not a claim about which arm
performs better -- the toy-sized comparison here is not the M3/M4 result;
a full walk (including the final test) would need ~47 retrains and
should be run separately once the group wants that number.

Usage:
    python scripts/run_lstm_quarterly_retrain_check.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, M2Calendar, initial_window_length  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec  # noqa: E402
from evaluation import sharpe_ratio, cagr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_data import train_lstm_variants, LSTM_N_STATES  # noqa: E402


def main() -> None:
    print("=" * 78)
    print("LSTM QUARTERLY RETRAINING CHECK -- validation period only (2015-2018)")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    full_returns = ds.returns

    calendar = M2Calendar()
    initial_window = initial_window_length(full_returns.index, calendar)
    dev_val_returns = full_returns.loc[:calendar.validation_end]
    print(f"\nInitial window: {initial_window} days (through {calendar.initial_training_end.date()})")
    print(f"Validation walk-forward: {len(dev_val_returns) - initial_window} days")

    co = ds.close_to_open_returns if ds.opens is not None else None
    oc = ds.open_to_close_returns if ds.opens is not None else None
    spec = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30)

    print("\n--- Training initial LSTM models on the initial window (shared starting point) ---")
    t0 = time.time()
    lstm_models_frozen, feature_matrix = train_lstm_variants(dev_val_returns, initial_window)
    print(f"Initial training runtime: {time.time() - t0:.1f}s\n")

    # A second, independent copy for the quarterly arm (retraining mutates
    # the model objects in place -- the frozen arm's models must stay untouched).
    lstm_models_quarterly, feature_matrix_q = train_lstm_variants(dev_val_returns, initial_window)

    print("--- frozen (today's default) ---")
    t0 = time.time()
    result_frozen = run_walk_forward(
        dev_val_returns, n_states=LSTM_N_STATES, initial_window=initial_window,
        constraint_spec=spec, close_to_open_returns=co, open_to_close_returns=oc,
        lstm_models=lstm_models_frozen, lstm_feature_matrix=feature_matrix,
    )
    print(f"runtime: {time.time() - t0:.1f}s, {len(result_frozen.lstm_retrain_dates)} retrains")

    print("\n--- quarterly retrain ---")
    t0 = time.time()
    result_quarterly = run_walk_forward(
        dev_val_returns, n_states=LSTM_N_STATES, initial_window=initial_window,
        constraint_spec=spec, close_to_open_returns=co, open_to_close_returns=oc,
        lstm_models=lstm_models_quarterly, lstm_feature_matrix=feature_matrix_q,
        lstm_retrain_every_n_rebalances=3, lstm_retrain_n_epochs=50,
    )
    print(f"runtime: {time.time() - t0:.1f}s, {len(result_quarterly.lstm_retrain_dates)} retrains")

    print("\n" + "=" * 78)
    print("VALIDATION-PERIOD RESULTS")
    print("=" * 78)
    for config in ["lstm_baseline", "lstm_regime", "lstm_blend"]:
        rf = result_frozen.portfolio_returns[config]
        rq = result_quarterly.portfolio_returns[config]
        print(f"{config:14} frozen: sharpe={sharpe_ratio(rf):.3f} cagr={cagr(rf):.4f}   "
              f"quarterly: sharpe={sharpe_ratio(rq):.3f} cagr={cagr(rq):.4f}")

    print("\nCapability check only -- see this script's module docstring. "
          "A full-walk (including final test) comparison is a separate, "
          "much longer run.")


if __name__ == "__main__":
    main()
