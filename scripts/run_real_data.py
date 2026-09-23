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

Initial window and reporting periods follow M2's calendar
(data.M2Calendar) rather than an arbitrary trading-day count: the
initial fit uses every trading day through 2014-12-31, walk-forward
days from 2015-01-01 through 2018-12-31 are the chronological
validation period, and only 2019-01-01 through 2026-08-31 is the final
test. Metrics are reported separately for validation and final test --
never pooled -- and anything after 2026-08-31 (an extended data pull
run after the M2 calendar was fixed) is reported separately again and
excluded from both, since scoring it as part of the "final test" would
retroactively widen a window that was supposed to be frozen.

Usage:
    python scripts/run_real_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, UNIVERSE, M2Calendar, initial_window_length, stage_labels  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec, drift_weights  # noqa: E402
from evaluation import (  # noqa: E402
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
)


def binding_constraints_dataframe(binding_history: list, calendar: M2Calendar, stage: str) -> pd.DataFrame:
    """Flatten one config's binding_constraints_history (per M2/M3's "we
    will report on which constraints are binding") into a row-per-rebalance
    table restricted to `stage` ('validation' or 'final_test'), with asset
    indices translated to tickers for readability."""
    rows = []
    for entry in binding_history:
        date = entry["date"]
        if stage_labels(pd.DatetimeIndex([date]), calendar).iloc[0] != stage:
            continue
        rows.append({
            "date": date,
            "turnover": round(entry["turnover"], 4),
            "turnover_binding": entry["turnover_binding"],
            "lower_bound_binding": ",".join(UNIVERSE[i] for i in entry["lower_bound_binding"]),
            "upper_bound_binding": ",".join(UNIVERSE[i] for i in entry["upper_bound_binding"]),
        })
    return pd.DataFrame(rows)


def summarize(result, returns: pd.DataFrame, initial_window: int, mask: np.ndarray) -> pd.DataFrame:
    """Metrics for the subset of walk-forward days selected by `mask`
    (boolean array aligned with result.dates), using each config's own
    drifted (pre-trade) weights as the turnover reference -- identical to
    average_turnover's definition elsewhere in the harness."""
    rows = []
    for config, r in result.portfolio_returns.items():
        w = result.weights[config]
        w_drift = np.array([
            drift_weights(w[i - 1] if i > 0 else w[0], np.exp(returns.to_numpy()[initial_window - 1 + i]))
            for i in range(len(w))
        ])
        rows.append({
            "config": config,
            "n_days": int(mask.sum()),
            "sharpe": round(sharpe_ratio(r[mask]), 3),
            "cagr": round(cagr(r[mask]), 4),
            "max_drawdown": round(max_drawdown(r[mask]), 4),
            "sortino": round(sortino_ratio(r[mask]), 3),
            "calmar": round(calmar_ratio(r[mask]), 3),
            "avg_turnover": round(average_turnover(w[mask], w_drift[mask]), 4),
            "concentration_hhi": round(herfindahl_concentration(w[mask]), 4),
        })
    return pd.DataFrame(rows).set_index("config")


def main() -> None:
    print("=" * 78)
    print("REAL-DATA RUN -- actual ten-ETF universe, 2007-04-11 onward")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    returns = ds.returns
    print(f"\nLoaded {len(returns)} trading days, {returns.index.min().date()} "
          f"to {returns.index.max().date()}, tickers: {list(returns.columns)}")

    calendar = M2Calendar()
    initial_window = initial_window_length(returns.index, calendar)
    print(f"Initial training window per M2's calendar: {initial_window} trading days "
          f"through {calendar.initial_training_end.date()}")

    t0 = time.time()
    result = run_walk_forward(
        returns,
        n_states=3,
        initial_window=initial_window,
        alpha=0.97,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    runtime = time.time() - t0
    print(f"\nWalk-forward runtime: {runtime:.1f}s over {len(result.dates)} trading days\n")

    stages = stage_labels(pd.DatetimeIndex(result.dates), calendar)
    stage_counts = stages.value_counts()
    for stage in ["validation", "final_test", "post_final_test"]:
        if stage not in stage_counts:
            stage_counts[stage] = 0

    for stage, heading in [
        ("validation", "VALIDATION (2015-01-01 -- 2018-12-31): development/tuning period, not a final-test result"),
        ("final_test", "FINAL TEST (2019-01-01 -- 2026-08-31): the frozen, one-shot out-of-sample result"),
        ("post_final_test", "AFTER M2's calendar (post 2026-08-31): exploratory only, excluded from the final-test figure"),
    ]:
        n = int(stage_counts[stage])
        print(f"--- {heading} [{n} days] ---")
        if n == 0:
            print("(none)\n")
            continue
        mask = (stages == stage).to_numpy()
        summary = summarize(result, returns, initial_window, mask)
        print(summary.to_string())
        if stage == "final_test":
            out_dir = Path(__file__).resolve().parents[1] / "outputs"
            out_dir.mkdir(exist_ok=True)
            summary.to_csv(out_dir / "real_data_summary.csv")
            print(f"Saved to {out_dir / 'real_data_summary.csv'}")
        print()

    final_test_mask = (stages == "final_test").to_numpy()
    if final_test_mask.any():
        print("--- Paired block-bootstrap on the final test: erc_blend vs erc_baseline (Sharpe difference) ---")
        ci = block_bootstrap_diff_ci(
            result.portfolio_returns["erc_blend"][final_test_mask],
            result.portfolio_returns["erc_baseline"][final_test_mask],
            n_boot=1000, random_state=0,
        )
        print(f"point estimate: {ci['point_estimate']:.3f}, "
              f"{int(ci['ci_level']*100)}% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    print("\n--- Reliability path pi_t (full walk-forward history) ---")
    pi = np.array(result.pi_history)
    print(f"mean pi_t: {pi.mean():.3f}  min: {pi.min():.3f}  max: {pi.max():.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    pd.Series(result.pi_history, index=result.dates, name="pi_t").to_csv(out_dir / "real_data_pi_path.csv")

    print("\n--- Binding constraints on final-test rebalances (M2/M3: 'we will "
          "report on which constraints are binding') ---")
    for config, binding_history in result.binding_constraints_history.items():
        table = binding_constraints_dataframe(binding_history, calendar, "final_test")
        n_turnover_binding = int(table["turnover_binding"].sum()) if len(table) else 0
        n_bound_binding = int((table["lower_bound_binding"].astype(bool) | table["upper_bound_binding"].astype(bool)).sum()) if len(table) else 0
        print(f"{config}: {len(table)} final-test rebalances, "
              f"{n_turnover_binding} with turnover binding, "
              f"{n_bound_binding} with a weight bound binding")
        table.to_csv(out_dir / f"real_data_binding_constraints_{config}.csv", index=False)
    print(f"Saved per-config detail to {out_dir}/real_data_binding_constraints_<config>.csv")
    print("\nThese are real-universe numbers, but still only the initial-window-fit, "
          "no-scheduled-refit, ERC-only pass documented in walkforward.py -- read that "
          "module's docstring before citing specific figures in M3/M4. Only the "
          "FINAL TEST block above is the number to cite; VALIDATION exists for "
          "development and must not be reported as an out-of-sample result.")


if __name__ == "__main__":
    main()
