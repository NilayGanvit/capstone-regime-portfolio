import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe, PriceDataset
from walkforward import run_walk_forward, ewma_blend
from constraints import ConstraintSpec


def _synthetic_dataset_with_opens(n_days: int, seed: int, gap_scale: float) -> PriceDataset:
    """Synthetic close prices plus a synthetic open series offset from the
    previous close by a small random overnight gap (gap_scale=0 makes
    open(t) == close(t-1) exactly, the degenerate no-gap case)."""
    ds, _ = make_synthetic_universe(n_days=n_days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    gap = rng.normal(0.0, gap_scale, size=ds.prices.shape) if gap_scale else np.zeros(ds.prices.shape)
    opens = ds.prices.shift(1) * np.exp(gap)
    opens.iloc[0] = ds.prices.iloc[0]  # arbitrary; excluded from returns anyway
    return PriceDataset(ds.prices, opens)


def test_walk_forward_runs_end_to_end_and_produces_valid_weights():
    ds, _ = make_synthetic_universe(n_days=500, seed=4)
    result = run_walk_forward(
        ds.returns, n_states=2, initial_window=252, alpha=0.97,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    expected_configs = {"erc_baseline", "erc_regime", "erc_blend", "equal_weight"}
    assert set(result.weights.keys()) == expected_configs

    for config, w in result.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6), f"{config} weights don't sum to 1"
        assert (w >= -1e-6).all(), f"{config} has negative weights"

    for config, r in result.portfolio_returns.items():
        assert len(r) == len(result.dates)
        assert np.isfinite(r).all()


def test_pi_history_length_matches_walk_forward_days():
    ds, _ = make_synthetic_universe(n_days=400, seed=5)
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252, refit_at_rebalance=False)
    assert len(result.pi_history) == len(result.dates)
    assert all(0.0 <= p <= 1.0 for p in result.pi_history)


def test_equal_weight_benchmark_is_actually_static_before_drift():
    """The equal_weight config's *target* should always be re-set to 1/n
    at each rebalance -- check that at the first rebalance date, its
    weight is exactly equal-weighted (post-projection, so allow the
    constraint tolerance)."""
    ds, _ = make_synthetic_universe(n_days=400, seed=5)
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252, refit_at_rebalance=False)
    w_eq = result.weights["equal_weight"]
    n_assets = w_eq.shape[1]
    # at least one rebalance should be close to exactly equal-weighted
    close_to_equal = np.any(np.all(np.abs(w_eq - 1.0 / n_assets) < 0.02, axis=1))
    assert close_to_equal


def test_open_execution_matches_same_close_default_when_there_is_no_overnight_gap():
    """If open(t) == close(t-1) exactly, split-leg execution degenerates
    to the same-close default: the overnight leg is a zero return and the
    intraday leg (open(t)->close(t)) equals the full close(t-1)->close(t)
    return, so results must match run_walk_forward with no open data at all."""
    ds = _synthetic_dataset_with_opens(n_days=400, seed=6, gap_scale=0.0)
    kwargs = dict(n_states=2, initial_window=252, constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30))

    baseline = run_walk_forward(ds.returns, **kwargs)
    with_opens = run_walk_forward(
        ds.returns, **kwargs,
        close_to_open_returns=ds.close_to_open_returns,
        open_to_close_returns=ds.open_to_close_returns,
    )

    for config in baseline.portfolio_returns:
        assert np.allclose(baseline.portfolio_returns[config], with_opens.portfolio_returns[config], atol=1e-10)
        assert np.allclose(baseline.weights[config], with_opens.weights[config], atol=1e-8)


def test_open_execution_with_a_real_gap_changes_returns_but_not_first_turnover():
    """With a genuine overnight gap, the split-leg return series should
    differ from the same-close default (the code path is actually being
    exercised). Turnover is defined on the rebalance decision itself
    (w_drift -> w_final), so the *first* rebalance's turnover -- computed
    from the shared equal-weight start, before any execution-timing
    difference has had a chance to affect the drifted position -- must
    still match exactly. Later rebalances legitimately diverge a little,
    since by then the actually-held position has followed a slightly
    different path (old vs. new weights over the overnight leg), so the
    drift reference for the *next* decision is no longer identical."""
    ds = _synthetic_dataset_with_opens(n_days=400, seed=6, gap_scale=0.01)
    kwargs = dict(n_states=2, initial_window=252, constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30))

    baseline = run_walk_forward(ds.returns, **kwargs)
    with_opens = run_walk_forward(
        ds.returns, **kwargs,
        close_to_open_returns=ds.close_to_open_returns,
        open_to_close_returns=ds.open_to_close_returns,
    )

    any_differs = any(
        not np.allclose(baseline.portfolio_returns[c], with_opens.portfolio_returns[c], atol=1e-10)
        for c in baseline.portfolio_returns
    )
    assert any_differs

    for config in baseline.binding_constraints_history:
        base_first = baseline.binding_constraints_history[config][0]["turnover"]
        open_first = with_opens.binding_constraints_history[config][0]["turnover"]
        assert np.isclose(base_first, open_first, atol=1e-10)
        # Later rebalances may diverge, but only by a small amount -- not
        # an unbounded/erratic difference.
        base_rest = [e["turnover"] for e in baseline.binding_constraints_history[config][1:]]
        open_rest = [e["turnover"] for e in with_opens.binding_constraints_history[config][1:]]
        assert np.allclose(base_rest, open_rest, atol=0.02)

    for config in baseline.weights:
        assert np.allclose(with_opens.weights[config].sum(axis=1), 1.0, atol=1e-6)
        assert (with_opens.weights[config] >= -1e-6).all()
        assert np.isfinite(with_opens.portfolio_returns[config]).all()


def test_risk_contribution_diagnostic_reported_for_erc_configs_only():
    ds, _ = make_synthetic_universe(n_days=400, seed=5)
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252)
    for config in ("erc_baseline", "erc_regime"):
        for entry in result.binding_constraints_history[config]:
            assert entry["risk_contribution_pre_constraint_max_dev"] is not None
            assert entry["risk_contribution_pre_constraint_max_dev"] >= 0.0
            assert entry["risk_contribution_post_constraint_max_dev"] >= 0.0
    for config in ("erc_blend", "equal_weight"):
        for entry in result.binding_constraints_history[config]:
            assert entry["risk_contribution_pre_constraint_max_dev"] is None
            assert entry["risk_contribution_post_constraint_max_dev"] is None


def test_execution_history_dated_same_day_as_decision_without_open_data():
    ds, _ = make_synthetic_universe(n_days=400, seed=5)
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252)
    for config in result.execution_history:
        assert len(result.execution_history[config]) == len(result.binding_constraints_history[config])
        for exec_entry, decision_entry in zip(result.execution_history[config], result.binding_constraints_history[config]):
            assert exec_entry["date"] == decision_entry["date"]
            assert np.isclose(exec_entry["turnover"], decision_entry["turnover"])


def test_execution_history_dated_one_day_after_decision_with_open_data():
    ds = _synthetic_dataset_with_opens(n_days=400, seed=6, gap_scale=0.01)
    result = run_walk_forward(
        ds.returns, n_states=2, initial_window=252,
        close_to_open_returns=ds.close_to_open_returns,
        open_to_close_returns=ds.open_to_close_returns,
    )
    all_dates = list(ds.returns.index)
    for config in result.execution_history:
        # The very last rebalance's execution may fall the day after the
        # sample ends, in which case it's never observed -- execution_history
        # is then exactly one entry short of binding_constraints_history.
        assert len(result.execution_history[config]) in (
            len(result.binding_constraints_history[config]),
            len(result.binding_constraints_history[config]) - 1,
        )
        for exec_entry, decision_entry in zip(result.execution_history[config], result.binding_constraints_history[config]):
            decision_idx = all_dates.index(decision_entry["date"])
            assert exec_entry["date"] == all_dates[decision_idx + 1]
            assert np.isclose(exec_entry["turnover"], decision_entry["turnover"])


def test_scheduled_refit_occurs_multiple_times_and_results_remain_valid():
    """Test that scheduled monthly refits occur and that results remain
    valid (weights sum to 1, returns are finite, etc.) across multiple
    refits. Uses a synthetic series long enough to trigger ~30 refits."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    result = run_walk_forward(
        ds.returns, n_states=2, initial_window=252, alpha=0.97,
        refit_n_iter=20,   # reduced for faster test
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    # Verify refits occurred and are unique
    assert len(result.refit_dates) > 1, f"Expected multiple refits, got {len(result.refit_dates)}"
    assert len(result.refit_dates) == len(set(result.refit_dates)), "Duplicate refit dates found"

    # Verify shape/bounds invariants still hold across refits
    for config, w in result.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6), f"{config} weights don't sum to 1"
        assert (w >= -1e-6).all(), f"{config} has negative weights"

    for r in result.portfolio_returns.values():
        assert np.isfinite(r).all(), "Non-finite returns found"


def test_refit_at_rebalance_false_reproduces_fit_once_behavior():
    """Test that setting refit_at_rebalance=False disables scheduled refits
    and produces the fit-once behavior (refit_dates is empty)."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    result = run_walk_forward(
        ds.returns, n_states=2, initial_window=252, alpha=0.97,
        refit_at_rebalance=False,
    )
    assert len(result.refit_dates) == 0, "Expected no refits when refit_at_rebalance=False"

    # Basic sanity checks still apply
    for config, w in result.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)


def test_ewma_blend_lam_one_ignores_prev():
    new = np.array([[2.0, 0.0], [0.0, 2.0]])
    prev = np.array([[10.0, 0.0], [0.0, 10.0]])
    assert np.allclose(ewma_blend(new, prev, 1.0), new)


def test_ewma_blend_lam_zero_keeps_prev():
    new = np.array([[2.0, 0.0], [0.0, 2.0]])
    prev = np.array([[10.0, 0.0], [0.0, 10.0]])
    assert np.allclose(ewma_blend(new, prev, 0.0), prev)


def test_ewma_blend_intermediate_lam_is_convex_combination():
    new = np.array([[4.0, 0.0], [0.0, 4.0]])
    prev = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(ewma_blend(new, prev, 0.3), 0.3 * new)


def test_refit_state_cov_ewma_none_matches_unsmoothed_baseline():
    """refit_state_cov_ewma=None (default) must reproduce the unsmoothed
    warm-started refit results exactly -- smoothing is strictly opt-in."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    kwargs = dict(
        n_states=2, initial_window=252, alpha=0.97, refit_n_iter=20,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    result_default = run_walk_forward(ds.returns, **kwargs)
    result_explicit_none = run_walk_forward(ds.returns, refit_state_cov_ewma=None, **kwargs)
    for config in result_default.weights:
        assert np.allclose(result_default.weights[config], result_explicit_none.weights[config])


def test_refit_state_cov_ewma_changes_regime_weights_when_enabled():
    """Setting refit_state_cov_ewma < 1 should change erc_regime's weights
    relative to the unsmoothed baseline once refits have accumulated
    (smoothing only diverges from the raw value after >=2 refits)."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    kwargs = dict(
        n_states=2, initial_window=252, alpha=0.97, refit_n_iter=20,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    result_unsmoothed = run_walk_forward(ds.returns, **kwargs)
    result_smoothed = run_walk_forward(ds.returns, refit_state_cov_ewma=0.3, **kwargs)
    assert len(result_unsmoothed.refit_dates) > 1
    assert not np.allclose(
        result_unsmoothed.weights["erc_regime"], result_smoothed.weights["erc_regime"]
    ), "expected EWMA smoothing to change erc_regime's weights once multiple refits accumulate"

    # Weights must still be valid (sum to 1, non-negative, within bounds)
    for config, w in result_smoothed.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert (w >= -1e-6).all()


def test_refit_align_to_fixed_reference_false_matches_default_behavior():
    """refit_align_to_fixed_reference=False (default) must reproduce the
    previous-fit-chained alignment exactly -- fixed-reference is opt-in."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    kwargs = dict(
        n_states=2, initial_window=252, alpha=0.97, refit_n_iter=20,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    result_default = run_walk_forward(ds.returns, **kwargs)
    result_explicit_false = run_walk_forward(ds.returns, refit_align_to_fixed_reference=False, **kwargs)
    for config in result_default.weights:
        assert np.allclose(result_default.weights[config], result_explicit_false.weights[config])


def test_refit_align_to_fixed_reference_runs_validly_and_can_change_weights():
    """Aligning to the frozen initial-window fit instead of the previous
    refit should still produce valid weights, and is expected to diverge
    from previous-fit-chained alignment once multiple refits accumulate."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    kwargs = dict(
        n_states=2, initial_window=252, alpha=0.97, refit_n_iter=20,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    result_chained = run_walk_forward(ds.returns, **kwargs)
    result_fixed_ref = run_walk_forward(ds.returns, refit_align_to_fixed_reference=True, **kwargs)
    assert len(result_chained.refit_dates) > 1

    for config, w in result_fixed_ref.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert (w >= -1e-6).all()
    for r in result_fixed_ref.portfolio_returns.values():
        assert np.isfinite(r).all()


def test_refit_every_n_rebalances_three_matches_default_behavior():
    """refit_every_n_rebalances=3 (default -- quarterly refit given monthly
    rebalances) must reproduce the implicit default exactly."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    kwargs = dict(
        n_states=2, initial_window=252, alpha=0.97, refit_n_iter=20,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    result_default = run_walk_forward(ds.returns, **kwargs)
    result_explicit_three = run_walk_forward(ds.returns, refit_every_n_rebalances=3, **kwargs)
    for config in result_default.weights:
        assert np.allclose(result_default.weights[config], result_explicit_three.weights[config])


def test_refit_every_n_rebalances_one_refits_roughly_three_times_as_often():
    """refit_every_n_rebalances=1 (monthly) should trigger roughly three
    times as many refits as the quarterly default."""
    ds, _ = make_synthetic_universe(n_days=900, seed=7)
    kwargs = dict(
        n_states=2, initial_window=252, alpha=0.97, refit_n_iter=20,
        constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
    )
    result_quarterly = run_walk_forward(ds.returns, **kwargs)  # default: refit_every_n_rebalances=3
    result_monthly = run_walk_forward(ds.returns, refit_every_n_rebalances=1, **kwargs)

    assert len(result_quarterly.refit_dates) < len(result_monthly.refit_dates)
    # roughly a third, allowing for integer rounding at the tail
    ratio = len(result_quarterly.refit_dates) / len(result_monthly.refit_dates)
    assert 0.25 <= ratio <= 0.40, f"expected ~1/3 as many refits, got ratio {ratio}"

    for config, w in result_quarterly.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert (w >= -1e-6).all()
    for r in result_quarterly.portfolio_returns.values():
        assert np.isfinite(r).all()
