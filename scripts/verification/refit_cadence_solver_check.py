"""
Does the SLSQP zero-weight failure on 2015-01-30 (and any other validation date)
change the validation-period HMM refit-cadence selection?

Repeats scripts/run_refit_cadence_validation_check.py (three arms: no refit,
monthly refit, quarterly refit; validation period only, the return series is cut
at the validation end) under either solver for the ERC raw target:

  --solver slsqp   the pipeline as frozen. Reproduces the saved comparison and
                   records, per arm, the dates on which the raw erc_regime or
                   erc_baseline target leaves an asset at zero weight.
  --solver exact   the same pipeline with the exact convex risk-budget solution.

The original selection rule (walkforward.py module docstring): an arm is retained
only if it matches or beats the no-refit baseline on erc_regime Sharpe. The saved
comparison selected quarterly on a margin of 0.004. This also reports that margin
with a paired stationary-bootstrap interval, because the margin is small whichever
solver is used.

Sharpe is reported two ways: gross with a zero risk-free rate (the basis the
original selection used) and in excess of the risk-free series.

Usage:
    python scripts/verification/refit_cadence_solver_check.py --solver slsqp
    python scripts/verification/refit_cadence_solver_check.py --solver exact
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import walkforward as wf  # noqa: E402
from allocation_erc import regularize_covariance, solve_risk_budget_convex  # noqa: E402
from constraints import ConstraintSpec  # noqa: E402
from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily  # noqa: E402
from evaluation import (  # noqa: E402
    cagr, max_drawdown, optimal_stationary_block_length, sharpe_ratio, stationary_bootstrap_indices,
)

ARCHIVE_DATA = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
ARMS = {
    "no_refit": dict(refit_at_rebalance=False),
    "monthly": dict(refit_at_rebalance=True, refit_warm_start=True, refit_every_n_rebalances=1),
    "quarterly": dict(refit_at_rebalance=True, refit_warm_start=True, refit_every_n_rebalances=3),
}
ZERO = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["slsqp", "exact"], required=True)
    args = parser.parse_args()

    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    calendar = M2Calendar()
    initial_window = initial_window_length(ds.returns.index, calendar)
    val_returns = ds.returns.loc[:calendar.validation_end]
    co = ds.close_to_open_returns if ds.opens is not None else None
    oc = ds.open_to_close_returns if ds.opens is not None else None

    raw_log = []  # (config, raw weights) in call order
    orig_base, orig_reg = wf.erc_baseline_weights, wf.erc_regime_weights

    def base(pooled_cov, shrinkage=0.10):
        w = (solve_risk_budget_convex(regularize_covariance(pooled_cov, shrinkage))
             if args.solver == "exact" else orig_base(pooled_cov, shrinkage))
        raw_log.append(("erc_baseline", w))
        return w

    def reg(state_covs, probs, shrinkage=0.10):
        w = (solve_risk_budget_convex(regularize_covariance(np.tensordot(probs, state_covs, axes=(0, 0)), shrinkage))
             if args.solver == "exact" else orig_reg(state_covs, probs, shrinkage))
        raw_log.append(("erc_regime", w))
        return w

    wf.erc_baseline_weights, wf.erc_regime_weights = base, reg
    rows, series, failures = [], {}, {}
    try:
        for arm, kwargs in ARMS.items():
            raw_log.clear()
            result = wf.run_walk_forward(
                val_returns, n_states=3, initial_window=initial_window, alpha=0.97,
                constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
                close_to_open_returns=co, open_to_close_returns=oc, **kwargs,
            )
            dates = [pd.Timestamp(e["date"]).date() for e in result.binding_constraints_history["erc_baseline"]]
            assert len(raw_log) == 2 * len(dates)
            failures[arm] = sorted({(dates[i // 2], name) for i, (name, w) in enumerate(raw_log) if (w < ZERO).any()})
            rf = load_risk_free_daily(str(ARCHIVE_DATA), pd.DatetimeIndex(result.dates))
            for config in ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]:
                r = result.portfolio_returns[config]
                rows.append({"solver": args.solver, "arm": arm, "config": config,
                             "sharpe_zero_rf": sharpe_ratio(r), "sharpe_excess": sharpe_ratio(r, risk_free=rf),
                             "cagr": cagr(r), "max_drawdown": max_drawdown(r)})
            series[arm] = result.portfolio_returns["erc_regime"]
    finally:
        wf.erc_baseline_weights, wf.erc_regime_weights = orig_base, orig_reg

    table = pd.DataFrame(rows)
    out = ROOT / "outputs" / f"refit_cadence_solver_check_{args.solver}.csv"
    table.to_csv(out, index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    print(f"\n=== solver: {args.solver} ===")
    print("SLSQP raw targets with an asset at zero weight, by arm:")
    for arm, f in failures.items():
        print(f"  {arm:10s} {[(str(d), n) for d, n in f] if f else 'none'}")
    reg_tab = table[table.config == "erc_regime"].set_index("arm")[["sharpe_zero_rf", "sharpe_excess", "cagr", "max_drawdown"]]
    print("\nerc_regime, validation 2015-2018:")
    print(reg_tab.round(4).to_string())

    base_s = reg_tab.loc["no_refit", "sharpe_zero_rf"]
    kept = [a for a in ["monthly", "quarterly"] if reg_tab.loc[a, "sharpe_zero_rf"] >= base_s]
    print(f"\nSelection rule (match or beat no_refit on erc_regime zero-rf Sharpe {base_s:.4f}): retains {kept or 'none'}")
    for a in ["monthly", "quarterly"]:
        print(f"  {a} minus no_refit: {reg_tab.loc[a, 'sharpe_zero_rf'] - base_s:+.4f}")

    rng = np.random.default_rng(0)
    for a in ["monthly", "quarterly"]:
        d = series[a] - series["no_refit"]
        block = optimal_stationary_block_length(d)
        stats = []
        for _ in range(2000):
            idx = stationary_bootstrap_indices(len(d), block, rng)
            stats.append(sharpe_ratio(series[a][idx]) - sharpe_ratio(series["no_refit"][idx]))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        print(f"  paired bootstrap, {a} - no_refit Sharpe: {sharpe_ratio(series[a]) - sharpe_ratio(series['no_refit']):+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (expected block {block:.1f})")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
