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

The Deflated Sharpe Ratio is reported as a disclosed sensitivity grid
(evaluation.dsr_sensitivity_grid), not one "primary" DSR, over both of
its uncertain inputs jointly: the cross-trial Sharpe variance and the
effective number of trials K. outputs/trial_candidate_sharpes.csv
recovers the per-candidate Sharpe behind every outputs/trial_log.csv
`counts_toward_dsr_trials=True` row (no daily return path survives for
any of them); evaluation.build_dsr_sensitivity_scenarios turns that
recovered set into several variance scenarios (pooled and per coherent
evaluation-sample family) and several K scenarios (literal logged
trial count, deduped candidate count, coherent search-family count),
following Lopez de Prado & Porcu (2026)'s framing that search-adjusted
significance depends jointly on effective trial count and cross-trial
dispersion, with neither fixed while the other varies. The conventional
unit-variance value is included as one scenario among several, not as
the default -- neither paper grounds it as a preferred fallback when
the actual cross-trial variance is uncertain, only as what the
original Bailey (2014) formula requires an input for.

Every grid cell reports two representations side by side (AnnaLisa,
2026-10-xx, against Lopez de Prado & Porcu 2026 pp.7-8): DSR-L
(evaluation.deflated_sharpe_ratio, unchanged -- search-adjusted
location mu_K plus the observed series' own skew/kurtosis-adjusted
sampling SE s_c) and DSR-LS (evaluation.deflated_sharpe_ratio_ls --
search-adjusted location AND scale (mu_K, sigma_K), the exact Gaussian
order-statistic moments of the search maximum via numerical
integration, no skew/kurtosis adjustment since sigma_K is a search-
distribution property, not a property of the observed series). DSR-EO
(the complete finite-sample search distribution) is not implemented --
it requires the full search distribution, which the surviving
candidate summary Sharpes cannot reconstruct.

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
    herfindahl_concentration, average_turnover,
    net_of_cost_returns, stationary_bootstrap_rq_contrasts,
    build_dsr_sensitivity_scenarios, dsr_sensitivity_grid,
)

FEE_BPS_SENSITIVITY = [5.0, 10.0, 25.0]  # M2's proposed baseline plus its two sensitivity checks
TRIAL_LOG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "trial_log.csv"
TRIAL_CANDIDATE_SHARPES_PATH = Path(__file__).resolve().parents[1] / "outputs" / "trial_candidate_sharpes.csv"
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
        ("validation", "VALIDATION (2015-01-01 -- 2018-12-31): development and specification-selection period"),
        ("final_test", "EXPLORATORY HISTORICAL WALK-FORWARD (2019-01-01 -- 2026-08-31): non-anticipative evaluation, but not an independent holdout because this period was previously inspected"),
        ("post_final_test", "POST-EVALUATION PERIOD (post 2026-08-31): exploratory only, reported separately"),
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
        print("--- Paired stationary-bootstrap inference for RQ1-RQ3 (5-bp net returns) ---")

        rq_config_names = [
            "erc_baseline",
            "erc_regime",
            "erc_blend",
            "lstm_baseline",
            "lstm_regime",
            "lstm_blend",
        ]
        rq_returns = {
            name: net_of_cost_returns(
                result.dates,
                result.portfolio_returns[name],
                result.execution_history[name],
                fee_bps=5.0,
            )[final_test_mask]
            for name in rq_config_names
        }

        rq_return_table = pd.DataFrame(
            {
                "date": pd.to_datetime(np.asarray(result.dates)[final_test_mask]),
                **{name: rq_returns[name] for name in rq_config_names},
            }
        )
        rq_returns_path = out_dir / "real_data_rq_net_returns_5bps.csv"
        rq_return_table.to_csv(rq_returns_path, index=False)
        print(f"Saved bootstrap input returns to {rq_returns_path}")

        rq_ci = stationary_bootstrap_rq_contrasts(
            rq_returns,
            n_boot=2000,
            ci=0.95,
            random_state=0,
        )

        labels = {
            "rq1_erc_regime_effect":
                "RQ1 ERC: regime - baseline",
            "rq1_lstm_regime_effect":
                "RQ1 LSTM: regime - baseline",
            "rq2_difference_in_differences":
                "RQ2 DiD: LSTM regime effect - ERC regime effect",
            "rq3_erc_reliability_effect":
                "RQ3 ERC: blend - regime",
            "rq3_lstm_reliability_effect":
                "RQ3 LSTM: blend - regime",
        }

        for key, label in labels.items():
            item = rq_ci[key]
            print(
                f"{label}: {item['point_estimate']:.3f}, "
                f"{int(item['ci_level'] * 100)}% CI "
                f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]"
            )

        meta = rq_ci["_bootstrap"]
        print(
            "Stationary bootstrap: "
            f"{meta['n_boot']} replications per contrast, "
            f"selection={meta['block_length_source']}, "
            f"synchronization={meta['synchronization']}"
        )
        for key, label in labels.items():
            print(
                f"  {label}: expected block length="
                f"{meta['contrast_block_lengths'][key]:.3f}"
            )

        rq_rows = []
        for key, label in labels.items():
            item = rq_ci[key]
            rq_rows.append({
                "contrast": key,
                "label": label,
                "point_estimate": item["point_estimate"],
                "ci_low": item["ci_low"],
                "ci_high": item["ci_high"],
                "ci_level": item["ci_level"],
                "fee_bps": 5.0,
                "n_boot": meta["n_boot"],
                "expected_block_length":
                    meta["contrast_block_lengths"][key],
                "block_length_source": meta["block_length_source"],
                "selector_series": meta["selector_series"],
                "synchronization": meta["synchronization"],
                "evaluation_start": "2019-01-01",
                "evaluation_end": "2026-08-31",
                "evaluation_status":
                    "exploratory_historical_walk_forward",
            })

        rq_table = pd.DataFrame(rq_rows)
        rq_path = out_dir / "real_data_rq_stationary_bootstrap.csv"
        rq_table.to_csv(rq_path, index=False)
        print(f"Saved to {rq_path}")

    if final_test_mask.any():
        print("\n--- Transaction costs: 5/10/25 bps sensitivity, exploratory historical walk-forward period ---")
        cost_table = cost_sensitivity_table(result, stages)
        print(cost_table.to_string())
        cost_table.to_csv(out_dir / "real_data_cost_sensitivity.csv")
        print(f"Saved to {out_dir / 'real_data_cost_sensitivity.csv'}")

        print("\n--- Transaction log (exploratory historical walk-forward period, 5 bps) ---")
        for config in result.execution_history:
            tx_log = transaction_log_dataframe(result.execution_history[config], calendar, "final_test", fee_bps=5.0)
            tx_log.to_csv(out_dir / f"real_data_transaction_log_{config}.csv", index=False)
        print(f"Saved per-config detail to {out_dir}/real_data_transaction_log_<config>.csv")

        print("\n--- Deflated Sharpe Ratio sensitivity grid, exploratory historical walk-forward period, net of 5 bps costs ---")
        print("No single (sharpe_variance_across_trials, n_trials) pair is treated as primary "
              "-- Lopez de Prado & Porcu (2026) frame search-adjusted significance as depending "
              "jointly on effective trial count AND cross-trial dispersion, and recommend "
              "reporting a disclosed range over defensible assumptions when dependence among "
              "candidates (here, several trials are sequential refinements of the same "
              "underlying run) cannot be estimated directly -- which is the case here since no "
              "daily return path survives for any historical trial. See "
              "outputs/trial_candidate_sharpes.csv for the recovered per-candidate Sharpes and "
              "evaluation.build_dsr_sensitivity_scenarios for how the grid below is assembled.")
        variance_scenarios, k_scenarios = build_dsr_sensitivity_scenarios(
            str(TRIAL_LOG_PATH), str(TRIAL_CANDIDATE_SHARPES_PATH),
        )
        print("\nVariance scenarios (cross-trial Sharpe variance):")
        for name, value in variance_scenarios.items():
            print(f"  {name}: {value:.4f}")
        print("K scenarios (effective number of trials):")
        for name, value in k_scenarios.items():
            print(f"  {name}: {value}")

        print("Each cell reports DSR-L (deflated_sharpe_ratio: search-adjusted location mu_K "
              "plus the observed series' own skew/kurtosis-adjusted sampling SE s_c) and DSR-LS "
              "(deflated_sharpe_ratio_ls: search-adjusted location AND scale (mu_K, sigma_K), a "
              "Gaussian-reference property of the search distribution itself, no skew/kurtosis "
              "adjustment) side by side -- two different representations of the same "
              "search-adjusted null, not two competing estimates of one quantity.")

        grid_rows = []
        for config in result.portfolio_returns:
            net_5bps = net_of_cost_returns(result.dates, result.portfolio_returns[config], result.execution_history[config], fee_bps=5.0)
            net_final = net_5bps[final_test_mask]
            sr = sharpe_ratio(net_final)
            grid = dsr_sensitivity_grid(
                observed_sharpe=sr, returns=net_final,
                variance_scenarios=variance_scenarios, k_scenarios=k_scenarios,
            )
            dsr_l_vals = [cell["dsr_l"] for row in grid.values() for cell in row.values()]
            dsr_ls_vals = [cell["dsr_ls"] for row in grid.values() for cell in row.values()]
            print(f"{config}: net-of-cost Sharpe={sr:.3f}, "
                  f"DSR-L range=[{min(dsr_l_vals):.3f}, {max(dsr_l_vals):.3f}], "
                  f"DSR-LS range=[{min(dsr_ls_vals):.3f}, {max(dsr_ls_vals):.3f}] "
                  f"across {len(variance_scenarios)}x{len(k_scenarios)} assumption grid")
            for v_name, row in grid.items():
                for k_name, cell in row.items():
                    grid_rows.append({
                        "config": config, "sharpe": round(sr, 4),
                        "variance_scenario": v_name, "variance_value": round(variance_scenarios[v_name], 4),
                        "k_scenario": k_name, "k_value": k_scenarios[k_name],
                        "dsr_l": round(cell["dsr_l"], 4),
                        "dsr_ls": round(cell["dsr_ls"], 4),
                    })
        pd.DataFrame(grid_rows).to_csv(out_dir / "real_data_dsr_sensitivity_grid.csv", index=False)
        print(f"Saved full grid to {out_dir / 'real_data_dsr_sensitivity_grid.csv'}")
        print("Compare each config's printed ranges above against the variance/K scenarios: on "
              "the archived reference run, every recovered empirical-variance scenario (pooled "
              "or per-family) lands far above the conventional unit-variance scenario for BOTH "
              "DSR-L and DSR-LS, regardless of which K is used -- a wide, config-independent "
              "swing driven almost entirely by the variance assumption. If that pattern holds "
              "here too, the DSR conclusion is NOT stable across this grid, which is itself the "
              "reportable finding, not any single cell.")

    print("\n--- Reliability path pi_t (full walk-forward history) ---")
    pi = np.array(result.pi_history)
    print(f"mean pi_t: {pi.mean():.3f}  min: {pi.min():.3f}  max: {pi.max():.3f}")

    pd.Series(result.pi_history, index=result.dates, name="pi_t").to_csv(out_dir / "real_data_pi_path.csv")

    print("\n--- Binding constraints on exploratory historical walk-forward rebalances ---")
    for config, binding_history in result.binding_constraints_history.items():
        table = binding_constraints_dataframe(binding_history, calendar, "final_test")
        n_turnover_binding = int(table["turnover_binding"].sum()) if len(table) else 0
        n_bound_binding = int((table["lower_bound_binding"].astype(bool) | table["upper_bound_binding"].astype(bool)).sum()) if len(table) else 0
        print(f"{config}: {len(table)} exploratory-period rebalances, "
              f"{n_turnover_binding} with turnover binding, "
              f"{n_bound_binding} with a weight bound binding")
        table.to_csv(out_dir / f"real_data_binding_constraints_{config}.csv", index=False)
    print(f"Saved per-config detail to {out_dir}/real_data_binding_constraints_<config>.csv")
    print("\nThe run reports all six primary configurations "
          "(ERC and LSTM, each baseline/regime/blend) plus equal_weight. "
          "The primary LSTM parameters are trained on the initial estimation window "
          "and remain fixed during the walk-forward evaluation, while the HMM, "
          "predictive-density models, and state covariance estimates are refit on "
          "their scheduled expanding-window cadence. The 2015-2018 period is used "
          "for development and specification selection. The 2019-2026 results are "
          "non-anticipative historical walk-forward evidence but are not an "
          "independent holdout because that period was previously inspected. "
          "real_data_summary.csv reports gross performance. Net-of-cost results "
          "under the 5, 10, and 25 basis-point assumptions are reported in "
          "real_data_cost_sensitivity.csv; DSR is reported separately above.")


if __name__ == "__main__":
    main()
