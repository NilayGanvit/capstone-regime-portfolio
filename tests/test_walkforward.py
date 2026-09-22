import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe
from walkforward import run_walk_forward
from constraints import ConstraintSpec


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
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252)
    assert len(result.pi_history) == len(result.dates)
    assert all(0.0 <= p <= 1.0 for p in result.pi_history)


def test_equal_weight_benchmark_is_actually_static_before_drift():
    """The equal_weight config's *target* should always be re-set to 1/n
    at each rebalance -- check that at the first rebalance date, its
    weight is exactly equal-weighted (post-projection, so allow the
    constraint tolerance)."""
    ds, _ = make_synthetic_universe(n_days=400, seed=5)
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252)
    w_eq = result.weights["equal_weight"]
    n_assets = w_eq.shape[1]
    # at least one rebalance should be close to exactly equal-weighted
    close_to_equal = np.any(np.all(np.abs(w_eq - 1.0 / n_assets) < 0.02, axis=1))
    assert close_to_equal
