"""
End-to-end walk-forward run on the REAL ten-ETF universe.

Companion to scripts/run_smoke_test.py, which only ever ran on synthetic
2-regime data because no real price history was available in the
original development sandbox. This script loads the actual daily
adjusted-close history for the M2 universe (SPY, EFA, EEM, IEF, TLT,
LQD, HYG, GLD, DBC, VNQ; 2007-04-11 onward -- HYG's inception date sets
the common start) from data/raw/, via data.PriceDataset.from_csv_dir,
and runs the same ERC baseline/regime/reliability-blend walk-forward
harness used by the smoke test, extended with the LSTM allocator so all
six M2 evaluation-table configs (+ equal_weight) come out of one run.

Before the walk-forward call, this script trains both LSTM variants
(baseline, regime) end-to-end (allocation_lstm.train_lstm_allocator) on
the initial training window ONLY -- never dev+validation like
scripts/run_lstm_training.py's convergence check, and never the final
test -- so that walk-forward's own 2015-2026 walk (including its
validation days) is scored against a model that has genuinely never seen
those days, exactly the same chronological discipline the HMM/M0/M1
initial fit already follows. The trained models are then passed into
run_walk_forward, which does inference only (no further training) at
every rebalance date -- see walkforward.py's module docstring for why
training and the harness stay separated. This is real-universe,
real-training, but still the initial-window-fit, no-scheduled-refit-of-
the-LSTM pass documented there -- the LSTM models are never retrained
mid-walk, unlike the HMM/M0/M1 which refit quarterly.

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

Transaction costs, DSR, and the trial log: run_walk_forward itself
charges no fee (weights don't depend on the fee rate), so this script
derives net-of-cost returns at 5/10/25 bps from one run via
evaluation.net_of_cost_returns and reports the 5 bps figure (M2's
proposed baseline) as the headline final-test number, with 10/25 bps as
a documented sensitivity table -- not three separate walk-forward runs.
The Deflated Sharpe Ratio uses the 5 bps net final-test Sharpe and
n_trials read from outputs/trial_log.csv (evaluation.n_trials_from_log),
which only counts rows recording an actual performance-driven
comparison, not plain correctness fixes -- see that file's own notes
column, including the still-open refit-cadence question flagged by
Silvio's M3 review (2026-09-25): quarterly refit was chosen by comparing
Sharpe over a sample that includes the designated final-test window, so
it is not yet a validation-only-derived default.

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
from features import build_feature_matrix  # noqa: E402
from regime import GaussianHMM  # noqa: E402
from allocation_lstm import RiskBudgetLSTM, build_lstm_training_windows, train_lstm_allocator  # noqa: E402
from walkforward import run_walk_forward  # noqa: E402
from constraints import ConstraintSpec, drift_weights  # noqa: E402
from evaluation import (  # noqa: E402
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
    net_of_cost_returns, deflated_sharpe_ratio, n_trials_from_log,
)

FEE_BPS_SENSITIVITY = [5.0, 10.0, 25.0]  # M2's proposed baseline plus its two sensitivity checks
TRIAL_LOG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "trial_log.csv"
LSTM_SEQ_LEN = 60
LSTM_COV_WINDOW = 252
LSTM_N_EPOCHS = 200
LSTM_N_STATES = 3  # matches this script's own n_states for run_walk_forward


def train_lstm_variants(returns: pd.DataFrame, initial_window: int) -> tuple[dict, pd.DataFrame]:
    """Train the baseline and regime-aware LSTM allocators on
    returns[:initial_window] only -- see this script's module docstring
    for why. Returns ({"baseline": model, "regime": model},
    full-history feature_matrix) ready to pass straight into
    run_walk_forward's lstm_models/lstm_feature_matrix."""
    n_assets = returns.shape[1]
    train_returns = returns.iloc[:initial_window]
    feature_matrix = build_feature_matrix(returns)  # causal by construction, safe over full history
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
        print(f"LSTM {variant_name}: {len(windows)} training windows, "
              f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}")
        model = RiskBudgetLSTM(n_features=windows[0].shape[1], n_assets=n_assets)
        t0 = time.time()
        result = train_lstm_allocator(
            model, windows, covs, fwd, n_epochs=LSTM_N_EPOCHS, turnover_penalty=0.1, shrinkage=0.10,
        )
        print(f"LSTM {variant_name}: trained in {time.time() - t0:.1f}s, "
              f"loss {result['loss_history'][0]:.4f} -> {result['loss_history'][-1]:.4f}")
        models[variant_name] = result["model"]
    return models, feature_matrix


def binding_constraints_dataframe(binding_history: list, calendar: M2Calendar, stage: str) -> pd.DataFrame:
    """Flatten one config's binding_constraints_history (per M2/M3's "we
    will report on which constraints are binding," plus "verify the
    degree of conformance... with the expected risk budgets") into a
    row-per-rebalance table restricted to `stage` ('validation' or
    'final_test'), with asset indices translated to tickers for
    readability."""
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
            "risk_contribution_pre_constraint_max_dev": entry["risk_contribution_pre_constraint_max_dev"],
            "risk_contribution_post_constraint_max_dev": entry["risk_contribution_post_constraint_max_dev"],
        })
    return pd.DataFrame(rows)


def transaction_log_dataframe(execution_history: list, calendar: M2Calendar, stage: str, fee_bps: float) -> pd.DataFrame:
    """Row-per-execution transaction log restricted to `stage`, per M2's
    "fees will be debited from portfolio equity and entered in the
    transaction log as a decrease in NAV." `date` here is the actual
    execution date (== decision date with no open-price data, decision
    date + 1 trading day once next-open execution is enabled), not the
    rebalance decision date."""
    rows = []
    for entry in execution_history:
        date = entry["date"]
        if stage_labels(pd.DatetimeIndex([date]), calendar).iloc[0] != stage:
            continue
        rows.append({
            "date": date,
            "turnover": round(entry["turnover"], 4),
            "fee_bps": fee_bps,
            "cost_as_fraction_of_nav": round(fee_bps / 10_000.0 * entry["turnover"], 6),
        })
    return pd.DataFrame(rows)


def cost_sensitivity_table(result, stages: pd.Series) -> pd.DataFrame:
    """Sharpe/CAGR at each of FEE_BPS_SENSITIVITY, final-test only, for
    every config -- derived from the same walk-forward run via
    net_of_cost_returns rather than three separate re-runs, since weights
    don't depend on the fee rate."""
    final_test_mask = (stages == "final_test").to_numpy()
    rows = []
    for config in result.portfolio_returns:
        gross = result.portfolio_returns[config]
        for fee_bps in FEE_BPS_SENSITIVITY:
            net = net_of_cost_returns(result.dates, gross, result.execution_history[config], fee_bps)
            rows.append({
                "config": config,
                "fee_bps": fee_bps,
                "sharpe_net": round(sharpe_ratio(net[final_test_mask]), 3),
                "cagr_net": round(cagr(net[final_test_mask]), 4),
            })
    return pd.DataFrame(rows).set_index(["config", "fee_bps"])


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

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    ds = PriceDataset.from_csv_dir(str(data_dir))
    returns = ds.returns
    print(f"\nLoaded {len(returns)} trading days, {returns.index.min().date()} "
          f"to {returns.index.max().date()}, tickers: {list(returns.columns)}")
    if ds.opens is not None:
        print("Open prices available -- rebalances execute at the next session's "
              "open per M2, not at the decision-day close.")
    else:
        print("WARNING: no open-price data found -- falling back to same-close "
              "execution timing (see data.PriceDataset.opens).")

    calendar = M2Calendar()
    initial_window = initial_window_length(returns.index, calendar)
    print(f"Initial training window per M2's calendar: {initial_window} trading days "
          f"through {calendar.initial_training_end.date()}")

    print(f"\n--- Training LSTM baseline/regime allocators on the initial "
          f"window only ({LSTM_N_EPOCHS} epochs each) ---")
    t0 = time.time()
    lstm_models, lstm_feature_matrix = train_lstm_variants(returns, initial_window)
    print(f"LSTM training runtime: {time.time() - t0:.1f}s\n")

    t0 = time.time()
    result = run_walk_forward(
        returns,
        n_states=LSTM_N_STATES,
        initial_window=initial_window,
        alpha=0.97,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
        close_to_open_returns=ds.close_to_open_returns if ds.opens is not None else None,
        open_to_close_returns=ds.open_to_close_returns if ds.opens is not None else None,
        lstm_models=lstm_models,
        lstm_feature_matrix=lstm_feature_matrix,
        lstm_seq_len=LSTM_SEQ_LEN,
    )
    runtime = time.time() - t0
    print(f"\nWalk-forward runtime: {runtime:.1f}s over {len(result.dates)} trading days, "
          f"{len(result.weights)} configs\n")

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

    if final_test_mask.any():
        print("\n--- Transaction costs: 5/10/25 bps sensitivity, final test only (M2's proposed baseline plus sensitivity checks) ---")
        cost_table = cost_sensitivity_table(result, stages)
        print(cost_table.to_string())
        cost_table.to_csv(out_dir / "real_data_cost_sensitivity.csv")
        print(f"Saved to {out_dir / 'real_data_cost_sensitivity.csv'}")

        print("\n--- Transaction log (final test, 5 bps -- M2's proposed baseline) ---")
        for config in result.execution_history:
            tx_log = transaction_log_dataframe(result.execution_history[config], calendar, "final_test", fee_bps=5.0)
            tx_log.to_csv(out_dir / f"real_data_transaction_log_{config}.csv", index=False)
        print(f"Saved per-config detail to {out_dir}/real_data_transaction_log_<config>.csv")

        print("\n--- Deflated Sharpe Ratio, final test, net of 5 bps costs ---")
        n_trials = n_trials_from_log(str(TRIAL_LOG_PATH))
        print(f"n_trials = {n_trials} (from {TRIAL_LOG_PATH.name}, performance-driven comparisons only -- see that file's notes column)")
        for config in result.portfolio_returns:
            net_5bps = net_of_cost_returns(result.dates, result.portfolio_returns[config], result.execution_history[config], fee_bps=5.0)
            net_final = net_5bps[final_test_mask]
            sr = sharpe_ratio(net_final)
            dsr = deflated_sharpe_ratio(observed_sharpe=sr, returns=net_final, n_trials=n_trials)
            print(f"{config}: net-of-cost Sharpe={sr:.3f}, DSR={dsr:.3f}")

    print("\n--- Reliability path pi_t (full walk-forward history) ---")
    pi = np.array(result.pi_history)
    print(f"mean pi_t: {pi.mean():.3f}  min: {pi.min():.3f}  max: {pi.max():.3f}")

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
    print("\nThese are real-universe numbers for all six M2 evaluation-table configs "
          "(ERC and LSTM, each baseline/regime/blend) plus equal_weight, from one "
          "harness run -- but the LSTM models are trained once on the initial window "
          "only and never retrained mid-walk, unlike the HMM/M0/M1 (see this script's "
          "and walkforward.py's module docstrings before citing specific figures in "
          "M3/M4). Only the FINAL TEST block above is the number to cite; VALIDATION "
          "exists for development and must not be reported as an out-of-sample result. "
          "real_data_summary.csv is GROSS of transaction costs -- the 5 bps net-of-cost "
          "figure (M2's proposed baseline) is in real_data_cost_sensitivity.csv and the "
          "DSR line above, not in real_data_summary.csv.")


if __name__ == "__main__":
    main()
