import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_erc import solve_risk_budget, risk_contributions, erc_regime_weights, regularize_covariance


def _realistic_cov(n=5, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.08, 0.22, n)
    corr = 0.3 * np.ones((n, n)) + 0.7 * np.eye(n)
    return np.outer(vols, vols) * corr


def test_erc_equalizes_risk_contributions():
    cov = _realistic_cov()
    w = solve_risk_budget(cov)
    rc = risk_contributions(w, cov)
    rc_share = rc / (w @ cov @ w)
    assert np.allclose(rc_share, 1.0 / len(w), atol=1e-3)


def test_erc_weights_are_long_only_and_budget_constrained():
    cov = _realistic_cov()
    w = solve_risk_budget(cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-8).all()


def test_arbitrary_target_budgets_are_respected():
    cov = _realistic_cov()
    n = cov.shape[0]
    target = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    w = solve_risk_budget(cov, target_budgets=target, w_bounds=(0.0, 1.0))
    rc = risk_contributions(w, cov)
    rc_share = rc / (w @ cov @ w)
    assert np.allclose(rc_share, target, atol=1e-2)


def test_regularize_covariance_preserves_trace_direction():
    cov = _realistic_cov()
    reg = regularize_covariance(cov, shrinkage=0.5)
    # shrinking toward a scaled identity should reduce off-diagonal magnitude
    assert np.abs(reg).sum() < np.abs(cov).sum()


def test_regime_weights_differ_across_very_different_state_covariances():
    n = 5
    calm = np.eye(n) * 0.0004
    stress = _realistic_cov(n) * 5
    state_covs = np.array([calm, stress])
    w_calm_dominant = erc_regime_weights(state_covs, regime_probs=np.array([0.99, 0.01]))
    w_stress_dominant = erc_regime_weights(state_covs, regime_probs=np.array([0.01, 0.99]))
    assert not np.allclose(w_calm_dominant, w_stress_dominant, atol=1e-3)
