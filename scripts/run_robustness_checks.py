"""
Pre-specified robustness checks (M3 "Scope and feasibility"): the 2-state
HMM alternative to the 3-state primary specification, and the no-VNQ /
no-HYG universe checks (each drops one ETF whose behavior could otherwise
dominate a single regime/state and make the headline result look like an
artifact of that one asset).

Companion to scripts/run_real_data.py, which only ever ran the primary
3-state, full-ten-ETF specification. This script re-runs the same ERC
baseline/regime/reliability-blend walk-forward harness under four
specifications on the same real data:

    primary_3state   : n_states=3, all ten ETFs (same as run_real_data.py)
    2state           : n_states=2, all ten ETFs
    no_vnq           : n_states=3, universe minus VNQ (listed real estate)
    no_hyg           : n_states=3, universe minus HYG (high-yield credit)

run_walk_forward already accepts n_states as a plain parameter and infers
n_assets from the returns DataFrame's column count rather than assuming
ten, so dropping a column from `returns` before calling it is enough to
run a robustness universe -- no changes to walkforward.py were needed.

This is still ERC-only (the LSTM allocator is not wired into
run_walk_forward; see that module's docstring), and still the
initial-window-fit-plus-scheduled-quarterly-refit configuration documented
there, not a claim that every other design choice has been stress-tested.

Usage:
    python scripts/run_robustness_checks.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, UNIVERSE  # noqa: E402
from regime import bic_for_state_counts  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec, drift_weights  # noqa: E402
from evaluation import (  # noqa: E402
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
)

SPECS = {
    "primary_3state": {"n_states": 3, "drop": None},
    "2state": {"n_states": 2, "drop": None},
    "no_vnq": {"n_states": 3, "drop": "VNQ"},
    "no_hyg": {"n_states": 3, "drop": "HYG"},
}


def summarize(result, returns: pd.DataFrame, initial_window: int) -> pd.DataFrame:
    rows = []
    for config, r in result.portfolio_returns.items():
        w = result.weights[config]
        w_drift = np.array([
            drift_weights(w[i - 1] if i > 0 else w[0], np.exp(returns.to_numpy()[initial_window - 1 + i]))
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
    return pd.DataFrame(rows).set_index("config")


def main() -> None:
    print("=" * 78)
    print("ROBUSTNESS CHECKS -- 2-state HMM, no-VNQ, no-HYG (M3 Scope and feasibility)")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    full_returns = ds.returns
    initial_window = 252

    print("\n--- BIC by candidate state count (initial 252-day window, full "
          "universe) -- context for the 2-state check below, per regime.py's "
          "bic_for_state_counts: 3-state is meant to be justified against 2 "
          "and 4, not on interpretability alone ---")
    bic_results = bic_for_state_counts(full_returns.to_numpy()[:initial_window], candidates=(2, 3, 4))
    for k, stats in bic_results.items():
        print(f"  n_states={k}: BIC={stats['bic']:.1f}  log_likelihood={stats['log_likelihood']:.1f}")

    all_summaries = {}
    regime_effect_rows = []

    for spec_name, spec in SPECS.items():
        drop = spec["drop"]
        returns = full_returns.drop(columns=[drop]) if drop else full_returns
        tickers = [t for t in UNIVERSE if t != drop]

        print(f"\n--- {spec_name} (n_states={spec['n_states']}, "
              f"universe={tickers if drop else 'full ten-ETF'}) ---")
        t0 = time.time()
        result = run_walk_forward(
            returns,
            n_states=spec["n_states"],
            initial_window=initial_window,
            alpha=0.97,
            constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
        )
        runtime = time.time() - t0
        print(f"runtime: {runtime:.1f}s over {len(result.dates)} trading days")

        summary = summarize(result, returns, initial_window)
        print(summary.to_string())
        all_summaries[spec_name] = summary

        regime_effect_rows.append({
            "spec": spec_name,
            "regime_effect_sharpe": round(summary.loc["erc_regime", "sharpe"] - summary.loc["erc_baseline", "sharpe"], 3),
            "reliability_effect_sharpe": round(summary.loc["erc_blend", "sharpe"] - summary.loc["erc_regime", "sharpe"], 3),
            "erc_baseline_sharpe": summary.loc["erc_baseline", "sharpe"],
            "erc_regime_sharpe": summary.loc["erc_regime", "sharpe"],
            "erc_blend_sharpe": summary.loc["erc_blend", "sharpe"],
        })

        print(f"--- {spec_name}: paired block-bootstrap, erc_regime vs erc_baseline (Sharpe difference) ---")
        ci = block_bootstrap_diff_ci(
            result.portfolio_returns["erc_regime"],
            result.portfolio_returns["erc_baseline"],
            n_boot=1000, random_state=0,
        )
        print(f"point estimate: {ci['point_estimate']:.3f}, "
              f"{int(ci['ci_level']*100)}% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    print("\n" + "=" * 78)
    print("Regime effect (erc_regime - erc_baseline) and reliability effect")
    print("(erc_blend - erc_regime), by specification -- is the primary")
    print("3-state/full-universe finding an artifact of that one choice?")
    print("=" * 78)
    effect_table = pd.DataFrame(regime_effect_rows).set_index("spec")
    print(effect_table.to_string())

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    for spec_name, summary in all_summaries.items():
        summary.to_csv(out_dir / f"robustness_{spec_name}_summary.csv")
    effect_table.to_csv(out_dir / "robustness_regime_effect_by_spec.csv")
    print(f"\nSaved per-spec summaries and {out_dir / 'robustness_regime_effect_by_spec.csv'}")
    print("\nThese are real-universe numbers under the pre-specified robustness")
    print("checks from the M2/M3 scope note, still under the initial-window-fit,")
    print("quarterly-refit, ERC-only pass documented in walkforward.py.")


if __name__ == "__main__":
    main()
