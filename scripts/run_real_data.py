"""
End-to-end walk-forward run on the REAL ten-ETF universe.

Companion to scripts/run_smoke_test.py, which only ever ran on synthetic
2-regime data because no real price history was available in the
original development sandbox. This script loads the actual daily
adjusted-close history for the M2 universe (SPY, EFA, EEM, IEF, TLT,
LQD, HYG, GLD, DBC, VNQ; 2007-04-11 onward -- HYG's inception date sets
the common start) from data/raw/, via data.PriceDataset.from_csv_dir,
and runs the same ERC baseline/regime/reliability-blend walk-forward
harness used by the smoke test.

This still wires ERC only, not the LSTM allocator (run_walk_forward
does not yet call into allocation_lstm.py; see walkforward.py's
module docstring on the scheduled-refit and LSTM-training scope
limitations, which apply here exactly as they do to the smoke test).
These are real numbers on the real universe, not a claim that the
walk-forward harness is feature-complete against the full M2 design.

Usage:
    python scripts/run_real_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec, drift_weights  # noqa: E402
from evaluation import (  # noqa: E402
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
)


def main() -> None:
    print("=" * 78)
    print("REAL-DATA RUN -- actual ten-ETF universe, 2007-04-11 onward")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    returns = ds.returns
    print(f"\nLoaded {len(returns)} trading days, {returns.index.min().date()} "
          f"to {returns.index.max().date()}, tickers: {list(returns.columns)}")

    t0 = time.time()
    result = run_walk_forward(
        returns,
        n_states=3,
        initial_window=252,
        alpha=0.97,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    runtime = time.time() - t0
    print(f"\nWalk-forward runtime: {runtime:.1f}s over {len(result.dates)} trading days\n")

    rows = []
    for config, r in result.portfolio_returns.items():
        w = result.weights[config]
        w_drift = np.array([
            drift_weights(w[i - 1] if i > 0 else w[0], np.exp(returns.to_numpy()[251 + i]))
            for i in range(len(w))
        ])
        rows.append({
            "config": config,
            "sharpe": round(sharpe_ratio(r), 3),
            "cagr": round(cagr(r), 4),
            "max_drawdown": round(max_drawdown(r), 4),
            "sortino": round(sortino_ratio(r), 3),
            "calmar": round(calmar_ratio(r), 3),
            "avg_turnover": round(average_turnover(w, w_drift), 4),
            "concentration_hhi": round(herfindahl_concentration(w), 4),
        })
    summary = pd.DataFrame(rows).set_index("config")
    print(summary.to_string())

    print("\n--- Paired block-bootstrap: erc_blend vs erc_baseline (Sharpe difference) ---")
    ci = block_bootstrap_diff_ci(
        result.portfolio_returns["erc_blend"],
        result.portfolio_returns["erc_baseline"],
        n_boot=1000, random_state=0,
    )
    print(f"point estimate: {ci['point_estimate']:.3f}, "
          f"{int(ci['ci_level']*100)}% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    print("\n--- Reliability path pi_t ---")
    pi = np.array(result.pi_history)
    print(f"mean pi_t: {pi.mean():.3f}  min: {pi.min():.3f}  max: {pi.max():.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "real_data_summary.csv")
    pd.Series(result.pi_history, index=result.dates, name="pi_t").to_csv(out_dir / "real_data_pi_path.csv")
    print(f"\nSaved summary to {out_dir / 'real_data_summary.csv'}")
    print("\nThese are real-universe numbers, but still only the initial-window-fit, "
          "no-scheduled-refit, ERC-only pass documented in walkforward.py -- read that "
          "module's docstring before citing specific figures in M3/M4.")


if __name__ == "__main__":
    main()
