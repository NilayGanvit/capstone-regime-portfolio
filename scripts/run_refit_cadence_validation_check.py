"""
Re-derive the HMM refit-cadence choice using ONLY the chronological
validation period (2015-01-01 -- 2018-12-31), per AnnaLisa's M3 review
request: the quarterly default (outputs/trial_log.csv, trial 6) was
originally selected by comparing Sharpe over a sample that included the
2019-2026-08 final-test window, which is exactly the kind of exploratory-
run contamination M2/M3's data-snooping discussion warns against.

This script never lets the final-test window enter the comparison at
all: the returns series itself is truncated to end at
data.M2Calendar.validation_end before being passed to run_walk_forward,
so even the scheduled refit's expanding window can't reach 2019+ data.
The resulting walk-forward days are exactly the validation period.

Reproduces the original comparison basis (see walkforward.py's module
docstring): "quarterly + warm-started refit was the only configuration
that matched or beat the no-refit baseline on erc_regime's Sharpe."
Three arms, same warm-started EM, same everything else:
    no_refit  : refit_at_rebalance=False (frozen initial-window fit)
    monthly   : refit_every_n_rebalances=1 (M2's specified cadence)
    quarterly : refit_every_n_rebalances=3 (current code default)

Usage:
    python scripts/run_refit_cadence_validation_check.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, M2Calendar, initial_window_length  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec  # noqa: E402
from evaluation import sharpe_ratio, cagr, max_drawdown  # noqa: E402

ARMS = {
    "no_refit": dict(refit_at_rebalance=False),
    "monthly": dict(refit_at_rebalance=True, refit_warm_start=True, refit_every_n_rebalances=1),
    "quarterly": dict(refit_at_rebalance=True, refit_warm_start=True, refit_every_n_rebalances=3),
}


def main() -> None:
    print("=" * 78)
    print("REFIT-CADENCE VALIDATION-ONLY CHECK -- 2015-01-01 to 2018-12-31")
    print("(final-test window never enters this comparison at all)")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    full_returns = ds.returns

    calendar = M2Calendar()
    initial_window = initial_window_length(full_returns.index, calendar)

    # Truncate the input series itself at validation_end -- the scheduled
    # refit's expanding window then physically cannot reach 2019+ data,
    # not just "we won't look at those days when reporting."
    dev_val_returns = full_returns.loc[:calendar.validation_end]
    walk_days = len(dev_val_returns) - initial_window
    print(f"\nInitial window: {initial_window} days (through {calendar.initial_training_end.date()})")
    print(f"Validation walk-forward: {walk_days} days "
          f"({dev_val_returns.index[initial_window].date()} to {dev_val_returns.index[-1].date()})")

    co = ds.close_to_open_returns if ds.opens is not None else None
    oc = ds.open_to_close_returns if ds.opens is not None else None

    rows = []
    for arm_name, kwargs in ARMS.items():
        print(f"\n--- {arm_name} ---")
        t0 = time.time()
        result = run_walk_forward(
            dev_val_returns,
            n_states=3,
            initial_window=initial_window,
            alpha=0.97,
            constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
            close_to_open_returns=co,
            open_to_close_returns=oc,
            **kwargs,
        )
        runtime = time.time() - t0
        n_refits = len(result.refit_dates)
        print(f"runtime: {runtime:.1f}s, {n_refits} refits over {len(result.dates)} validation days")
        for config in ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]:
            r = result.portfolio_returns[config]
            rows.append({
                "arm": arm_name,
                "config": config,
                "n_refits": n_refits,
                "sharpe": round(sharpe_ratio(r), 3),
                "cagr": round(cagr(r), 4),
                "max_drawdown": round(max_drawdown(r), 4),
            })

    table = pd.DataFrame(rows).set_index(["arm", "config"])
    print("\n" + "=" * 78)
    print("VALIDATION-ONLY RESULTS (2015-2018, never touches final test)")
    print("=" * 78)
    print(table.to_string())

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "refit_cadence_validation_comparison.csv")
    print(f"\nSaved to {out_dir / 'refit_cadence_validation_comparison.csv'}")

    baseline_sharpe = table.loc[("no_refit", "erc_regime"), "sharpe"]
    monthly_sharpe = table.loc[("monthly", "erc_regime"), "sharpe"]
    quarterly_sharpe = table.loc[("quarterly", "erc_regime"), "sharpe"]
    print(f"\nerc_regime Sharpe -- no_refit: {baseline_sharpe}, monthly: {monthly_sharpe}, "
          f"quarterly: {quarterly_sharpe}")
    print("Per the original comparison basis (walkforward.py's module docstring): "
          "does monthly or quarterly match/beat the no-refit baseline here, using "
          "validation-only evidence -- not the criterion originally used to pick quarterly.")


if __name__ == "__main__":
    main()
