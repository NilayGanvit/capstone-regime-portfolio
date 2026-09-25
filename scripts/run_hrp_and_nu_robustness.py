"""
Secondary robustness checks (M2's own scope note: "HRP and per-regime nu
are out of scope for this pass ... secondary robustness checks only if
time allows" -- see README "Known scope limitations"): does ERC's regime
effect survive under Hierarchical Risk Parity's tree-based construction
(allocation_hrp.py) instead of ERC's SLSQP risk-budgeting solve, and does
letting the M1 mixture's states have their own degrees-of-freedom
(densities.fit_per_regime_nu) instead of one shared nu change the
reliability layer's conclusion?

Companion to scripts/run_robustness_checks.py (which stress-tests the
2-state/no-VNQ/no-HYG dimensions on ERC). This script instead holds the
primary 3-state/full-universe specification fixed and varies two
different opt-in toggles on run_walk_forward:

    hrp_check     : same walk-forward run, include_hrp=True, comparing
                    HRP's regime effect (hrp_regime - hrp_baseline) and
                    reliability effect (hrp_blend - hrp_regime) against
                    ERC's own, on the identical dates and covariances.
    per_regime_nu_check : two walk-forward runs, per_regime_nu=False (the
                    shared-nu default) vs True, comparing ERC's regime/
                    reliability effect across the two.

Both toggles are additive/independent of everything else the primary run
already does (calendar-boundary reporting, next-session-open execution,
quarterly scheduled refit) -- see walkforward.py's module docstring.

Usage:
    python scripts/run_hrp_and_nu_robustness.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, M2Calendar, initial_window_length, stage_labels  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec, drift_weights  # noqa: E402
from evaluation import sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio  # noqa: E402

N_STATES = 3  # primary specification, held fixed for both checks


def summarize(result, returns: pd.DataFrame, initial_window: int, mask: np.ndarray) -> pd.DataFrame:
    """Final-test-only metrics (mask selects that stage's days), matching
    run_robustness_checks.py's convention -- see that script for why
    validation and final-test are never pooled into one figure."""
    rows = []
    for config, r in result.portfolio_returns.items():
        w = result.weights[config]
        w_drift = np.array([
            drift_weights(w[i - 1] if i > 0 else w[0], np.exp(returns.to_numpy()[initial_window - 1 + i]))
            for i in range(len(w))
        ])
        rows.append({
            "config": config,
            "sharpe": round(sharpe_ratio(r[mask]), 3),
            "cagr": round(cagr(r[mask]), 4),
            "max_drawdown": round(max_drawdown(r[mask]), 4),
            "sortino": round(sortino_ratio(r[mask]), 3),
            "calmar": round(calmar_ratio(r[mask]), 3),
        })
    return pd.DataFrame(rows).set_index("config")


def run_hrp_check(
    returns: pd.DataFrame, initial_window: int, calendar: M2Calendar,
    close_to_open_returns: pd.DataFrame | None, open_to_close_returns: pd.DataFrame | None,
) -> pd.DataFrame:
    print("=" * 78)
    print("HRP ROBUSTNESS CHECK -- does ERC's regime effect survive under")
    print("Hierarchical Risk Parity's tree-based construction? (M2 scope note)")
    print("=" * 78)

    t0 = time.time()
    result = run_walk_forward(
        returns,
        n_states=N_STATES,
        initial_window=initial_window,
        alpha=0.97,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
        close_to_open_returns=close_to_open_returns,
        open_to_close_returns=open_to_close_returns,
        include_hrp=True,
    )
    print(f"runtime: {time.time() - t0:.1f}s over {len(result.dates)} trading days")

    stages = stage_labels(pd.DatetimeIndex(result.dates), calendar)
    final_test_mask = (stages == "final_test").to_numpy()
    summary = summarize(result, returns, initial_window, final_test_mask)
    print(f"\nFINAL TEST ({calendar.validation_end.date()} < date <= {calendar.final_test_end.date()}, "
          f"{int(final_test_mask.sum())} days):")
    print(summary.to_string())

    effect_row = {
        "erc_regime_effect_sharpe": round(summary.loc["erc_regime", "sharpe"] - summary.loc["erc_baseline", "sharpe"], 3),
        "hrp_regime_effect_sharpe": round(summary.loc["hrp_regime", "sharpe"] - summary.loc["hrp_baseline", "sharpe"], 3),
        "erc_reliability_effect_sharpe": round(summary.loc["erc_blend", "sharpe"] - summary.loc["erc_regime", "sharpe"], 3),
        "hrp_reliability_effect_sharpe": round(summary.loc["hrp_blend", "sharpe"] - summary.loc["hrp_regime", "sharpe"], 3),
    }
    print("\n--- Regime/reliability effect: ERC vs HRP, same dates and covariances ---")
    for k, v in effect_row.items():
        print(f"  {k}: {v}")

    return summary, pd.DataFrame([effect_row])


def run_per_regime_nu_check(
    returns: pd.DataFrame, initial_window: int, calendar: M2Calendar,
    close_to_open_returns: pd.DataFrame | None, open_to_close_returns: pd.DataFrame | None,
) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("PER-REGIME NU ROBUSTNESS CHECK -- does letting each HMM state have")
    print("its own M1 degrees-of-freedom change the reliability layer's")
    print("conclusion, vs. one nu shared across every state? (M2 scope note)")
    print("=" * 78)

    rows = []
    summaries = {}
    for per_regime_nu in (False, True):
        label = "per_regime_nu" if per_regime_nu else "shared_nu"
        t0 = time.time()
        result = run_walk_forward(
            returns,
            n_states=N_STATES,
            initial_window=initial_window,
            alpha=0.97,
            constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
            close_to_open_returns=close_to_open_returns,
            open_to_close_returns=open_to_close_returns,
            per_regime_nu=per_regime_nu,
        )
        print(f"\n--- {label} -- runtime: {time.time() - t0:.1f}s over {len(result.dates)} trading days ---")

        stages = stage_labels(pd.DatetimeIndex(result.dates), calendar)
        final_test_mask = (stages == "final_test").to_numpy()
        summary = summarize(result, returns, initial_window, final_test_mask)
        print(summary.to_string())
        summaries[label] = summary

        rows.append({
            "spec": label,
            "erc_regime_effect_sharpe": round(summary.loc["erc_regime", "sharpe"] - summary.loc["erc_baseline", "sharpe"], 3),
            "erc_reliability_effect_sharpe": round(summary.loc["erc_blend", "sharpe"] - summary.loc["erc_regime", "sharpe"], 3),
        })

    effect_table = pd.DataFrame(rows).set_index("spec")
    print("\n--- Regime/reliability effect: shared nu vs per-regime nu ---")
    print(effect_table.to_string())
    return summaries, effect_table


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    returns = ds.returns
    # Next-session-open execution when open-price data is available --
    # same convention as scripts/run_robustness_checks.py and
    # scripts/run_real_data.py, so these checks are comparable to the
    # primary run's own execution timing rather than silently falling
    # back to same-close.
    co = ds.close_to_open_returns if ds.opens is not None else None
    oc = ds.open_to_close_returns if ds.opens is not None else None
    calendar = M2Calendar()
    initial_window = initial_window_length(returns.index, calendar)
    print(f"Initial training window per M2's calendar: {initial_window} trading days "
          f"through {calendar.initial_training_end.date()}\n")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)

    hrp_summary, hrp_effect = run_hrp_check(returns, initial_window, calendar, co, oc)
    hrp_summary.to_csv(out_dir / "robustness_hrp_summary.csv")
    print(f"\nSaved {out_dir / 'robustness_hrp_summary.csv'}")

    nu_summaries, nu_effect = run_per_regime_nu_check(returns, initial_window, calendar, co, oc)
    nu_effect.to_csv(out_dir / "robustness_per_regime_nu_summary.csv")
    print(f"Saved {out_dir / 'robustness_per_regime_nu_summary.csv'}")

    print("\nThese are real-universe, FINAL-TEST-ONLY numbers under the two")
    print("secondary robustness checks M2's own scope note deferred (HRP,")
    print("per-regime nu) -- still under the quarterly-refit pass documented")
    print("in walkforward.py, and neither check adds a new primary allocator")
    print("or density: HRP is compared against ERC as a robustness check on")
    print("the risk-budgeting construction, and per-regime nu is compared")
    print("against the shared-nu default as a robustness check on the M1")
    print("density, per README 'Known scope limitations.'")


if __name__ == "__main__":
    main()
