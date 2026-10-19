"""
15-seed robustness check of the LSTM regime effect under the frozen
bounded-Softmax, 5% floor specification, on the 2019-2026 final-test
period.

Context: an external review of the M6 draft raised two linked gaps,
both downstream of src/allocation_lstm.py:bounded_softmax and
scripts/run_real_data.py's LSTM_BUDGET_FLOOR=0.05 wiring:

  1. Tables 5/6 in the M6 draft are sourced from ordinary-Softmax,
     single-seed runs (baf2f0e Mac for Table 5, the Colab reference run
     for Table 6) -- neither reflects the proposed bounded-Softmax,
     5%-floor specification. (Addressed separately by wiring
     budget_floor into run_real_data.py's train_lstm_variants.)
  2. The headline "+0.26 LSTM regime effect" (RQ1's stationary-bootstrap
     contrast) comes from ONE seed on the final-test period. Table 4's
     ten-seed, 200-epoch experiment characterizes weight-budget
     DISPERSION under bounded Softmax, but only on the 2015-2018
     VALIDATION period, and never computes a Sharpe ratio at all. This
     script is the missing link: it runs the frozen bounded-Softmax,
     5%-floor spec across a seed sample on the FINAL TEST period, and
     reports the regime effect's distribution plus floor concentration
     and drawdown together.

An earlier 10-seed pass (seeds 0-9) already showed the regime effect
averaging well under the single-seed +0.26 headline. This is a smaller,
faster first pass on a different seed sample (3 fixed values plus 12
generated) before committing to a larger run, to check whether the
lower estimate holds up on seeds not drawn from a sequential range.

SEEDS has two parts: FIXED_SEEDS (3 specific values), then 12 more
generated round-robin from 2 separate RNGs (6 draws each), each seeded by one of
POPULAR_RNG_SEEDS (documented/culturally common values -- 42 and 3407
-- used because no single canonical "most popular seeds" ranking
exists; see the sources checked when this was first built). Using two
differently-seeded generators rather than one avoids the generated
portion being an artifact of any single RNG's stream.

This is NOT a performance-driven seed selection (contrast trial 12,
which selects run_real_data.py's own LSTM_RANDOM_STATE from a
VALIDATION-only comparison). Nothing here chooses a seed to use anywhere
else; it only characterizes the dispersion of an already-frozen spec
on the final-test period, so it does not count toward the DSR trial log
as a search step -- see outputs/trial_log.csv.

Floor concentration is measured on the RAW LSTM softmax budget output
(walkforward.py's new lstm_budgets_history diagnostic), i.e. the exact
quantity Table 4's "71% of baseline budgets ... were at the minimum"
statistic refers to -- not the post-risk-budgeting portfolio weight.

Mac environment, not Colab -- see AGENTS.md on LSTM run-to-run
environment sensitivity. This is a dispersion/robustness disclosure
(like Table 4, which was also run locally), not a replacement for the
Colab-canonical Table 5/6 headline numbers.

Usage:
    python scripts/run_lstm_bounded_softmax_robustness.py
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import PriceDataset, M2Calendar, initial_window_length, stage_labels  # noqa: E402
from features import build_feature_matrix  # noqa: E402
from regime import GaussianHMM  # noqa: E402
from allocation_lstm import RiskBudgetLSTM, build_lstm_training_windows, train_lstm_allocator  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec  # noqa: E402
from evaluation import sharpe_ratio, cagr, max_drawdown, net_of_cost_returns  # noqa: E402

FIXED_SEEDS = [123, 4, 2025]
POPULAR_RNG_SEEDS = [42, 3407]  # each seeds its own generator below
N_GENERATED = 12  # additional seeds beyond FIXED_SEEDS (6 per generator), not the total

_generators = [random.Random(s) for s in POPULAR_RNG_SEEDS]
_seen = set(FIXED_SEEDS)
_generated = []
_i = 0
while len(_generated) < N_GENERATED:
    candidate = _generators[_i % len(_generators)].randint(1, 10_000_000)
    if candidate not in _seen:
        _generated.append(candidate)
        _seen.add(candidate)
    _i += 1
SEEDS = FIXED_SEEDS + _generated
BUDGET_FLOOR = 0.05  # the frozen proposed specification
LSTM_SEQ_LEN = 60
LSTM_COV_WINDOW = 252
LSTM_N_EPOCHS = 200
LSTM_N_STATES = 3
HEADLINE_FEE_BPS = 5.0
FLOOR_TOLERANCE = 1e-4  # a budget within this of BUDGET_FLOOR counts as "at the floor"


def train_lstm_variants(returns: pd.DataFrame, initial_window: int, random_state: int) -> tuple[dict, pd.DataFrame]:
    """Same construction as scripts/run_real_data.py's train_lstm_variants
    (with budget_floor wired in), duplicated here so this seed sweep
    stays self-contained -- same pattern as the existing (uncommitted)
    run_lstm_seed_robustness.py."""
    n_assets = returns.shape[1]
    train_returns = returns.iloc[:initial_window]
    feature_matrix = build_feature_matrix(returns)
    train_features = feature_matrix.loc[:train_returns.index[-1]]

    hmm = GaussianHMM(n_states=LSTM_N_STATES, random_state=0).fit(train_returns.to_numpy())
    regime_probs = hmm.predicted_probabilities(train_returns.to_numpy())
    regime_prob_df = pd.DataFrame(
        regime_probs, index=train_returns.index,
        columns=[f"regime_prob_{k}" for k in range(LSTM_N_STATES)],
    )
    regime_features_train = train_features.join(regime_prob_df, how="inner")

    models = {}
    for variant_name, feats in [("baseline", train_features), ("regime", regime_features_train)]:
        windows, covs, fwd, dates = build_lstm_training_windows(
            feats, train_returns, seq_len=LSTM_SEQ_LEN, cov_window=LSTM_COV_WINDOW,
        )
        model = RiskBudgetLSTM(
            n_features=windows[0].shape[1], n_assets=n_assets,
            random_state=random_state, budget_floor=BUDGET_FLOOR,
        )
        result = train_lstm_allocator(
            model, windows, covs, fwd, n_epochs=LSTM_N_EPOCHS, turnover_penalty=0.1, shrinkage=0.10,
        )
        models[variant_name] = result["model"]
    return models, feature_matrix


def floor_concentration(budgets_history: list, final_test_dates: set) -> float:
    """Fraction of (rebalance x asset) raw-budget entries at the floor,
    restricted to final-test rebalances -- same statistic as Table 4's
    "71%/72% of budgets were at the minimum," computed on this seed's
    final-test window rather than Table 4's validation window."""
    entries = [
        b for row in budgets_history if row["date"] in final_test_dates
        for b in row["budgets"]
    ]
    if not entries:
        return float("nan")
    arr = np.array(entries)
    return float(np.mean(arr <= BUDGET_FLOOR + FLOOR_TOLERANCE))


def main() -> None:
    print("=" * 78)
    print("LSTM BOUNDED-SOFTMAX ROBUSTNESS -- FINAL TEST (2019-01-01 -- 2026-08-31)")
    print(f"budget_floor={BUDGET_FLOOR}, seeds={SEEDS}, {LSTM_N_EPOCHS} epochs each")
    print("=" * 78)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    full_returns = ds.returns
    calendar = M2Calendar()
    initial_window = initial_window_length(full_returns.index, calendar)
    co = ds.close_to_open_returns if ds.opens is not None else None
    oc = ds.open_to_close_returns if ds.opens is not None else None

    rows = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()
        lstm_models, lstm_feature_matrix = train_lstm_variants(full_returns, initial_window, random_state=seed)
        print(f"  trained in {time.time() - t0:.1f}s")

        t0 = time.time()
        result = run_walk_forward(
            full_returns,
            n_states=LSTM_N_STATES,
            initial_window=initial_window,
            alpha=0.97,
            constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
            close_to_open_returns=co,
            open_to_close_returns=oc,
            lstm_models=lstm_models,
            lstm_feature_matrix=lstm_feature_matrix,
            lstm_seq_len=LSTM_SEQ_LEN,
        )
        print(f"  walk-forward in {time.time() - t0:.1f}s over {len(result.dates)} days")

        stages = stage_labels(pd.DatetimeIndex(result.dates), calendar)
        final_test_mask = (stages == "final_test").to_numpy()
        final_test_dates = set(pd.DatetimeIndex(result.dates)[final_test_mask])

        metrics = {}
        for config in ["lstm_baseline", "lstm_regime", "lstm_blend"]:
            gross = result.portfolio_returns[config][final_test_mask]
            net = net_of_cost_returns(
                np.array(result.dates)[final_test_mask], gross,
                [e for e in result.execution_history[config] if e["date"] in final_test_dates],
                fee_bps=HEADLINE_FEE_BPS,
            )
            metrics[config] = {
                "sharpe_net": sharpe_ratio(net),
                "cagr_net": cagr(net),
                "max_drawdown_net": max_drawdown(net),
            }

        regime_effect = metrics["lstm_regime"]["sharpe_net"] - metrics["lstm_baseline"]["sharpe_net"]
        floor_baseline = floor_concentration(result.lstm_budgets_history["baseline"], final_test_dates)
        floor_regime = floor_concentration(result.lstm_budgets_history["regime"], final_test_dates)

        row = {"seed": seed, "regime_effect_sharpe": round(regime_effect, 4),
               "floor_concentration_baseline": round(floor_baseline, 4),
               "floor_concentration_regime": round(floor_regime, 4)}
        for config in ["lstm_baseline", "lstm_regime", "lstm_blend"]:
            row[f"{config}_sharpe_net"] = round(metrics[config]["sharpe_net"], 4)
            row[f"{config}_cagr_net"] = round(metrics[config]["cagr_net"], 4)
            row[f"{config}_max_drawdown_net"] = round(metrics[config]["max_drawdown_net"], 4)
        rows.append(row)
        print(f"  regime effect (Sharpe): {regime_effect:+.4f}  "
              f"floor conc. baseline/regime: {floor_baseline:.1%}/{floor_regime:.1%}")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("PER-SEED RESULTS")
    print("=" * 78)
    print(table.to_string(index=False))

    summary = table[["regime_effect_sharpe", "floor_concentration_baseline", "floor_concentration_regime",
                      "lstm_baseline_sharpe_net", "lstm_regime_sharpe_net",
                      "lstm_regime_max_drawdown_net"]].agg(["mean", "std", "min", "max"])
    print("\n" + "=" * 78)
    print("CROSS-SEED SUMMARY (final test, net of 5 bps, bounded Softmax 5% floor)")
    print("=" * 78)
    print(summary.to_string())

    n_positive = int((table["regime_effect_sharpe"] > 0).sum())
    print(f"\nRegime effect positive in {n_positive}/{len(SEEDS)} seeds "
          f"(mean {table['regime_effect_sharpe'].mean():+.4f}, "
          f"sd {table['regime_effect_sharpe'].std():.4f}).")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "lstm_bounded_softmax_robustness_15seed_raw.csv", index=False)
    summary.to_csv(out_dir / "lstm_bounded_softmax_robustness_15seed_summary.csv")
    print(f"\nSaved per-seed rows to {out_dir / 'lstm_bounded_softmax_robustness_15seed_raw.csv'}")
    print(f"Saved cross-seed summary to {out_dir / 'lstm_bounded_softmax_robustness_15seed_summary.csv'}")


if __name__ == "__main__":
    main()
