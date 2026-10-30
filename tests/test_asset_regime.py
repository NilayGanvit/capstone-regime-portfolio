import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_regime import asset_regime_covariance, asset_regime_forecast, fit_asset_hmms, mixture_variance
from constraints import ConstraintSpec
from data import make_synthetic_universe
from walkforward import run_walk_forward


def test_mixture_variance_matches_the_formula_and_ignores_state_labels():
    p = np.array([0.3, 0.7])
    mu = np.array([0.1, -0.2])
    var = np.array([1.0, 4.0])
    expected = 0.3 * (1.0 + 0.01) + 0.7 * (4.0 + 0.04) - (0.03 - 0.14) ** 2
    assert np.isclose(mixture_variance(p, mu, var), expected)
    assert np.isclose(mixture_variance(p[::-1], mu[::-1], var[::-1]), expected)
    assert np.isclose(mixture_variance(np.array([1.0, 0.0]), mu, var), 1.0)


def test_asset_regime_covariance_keeps_correlations_and_sets_the_diagonal():
    rng = np.random.default_rng(0)
    pooled = np.cov(rng.normal(size=(500, 4)).T)
    variances = np.array([1.0, 2.0, 0.5, 3.0])
    cov = asset_regime_covariance(pooled, variances)
    assert np.allclose(np.diag(cov), variances)
    sd_p, sd_c = np.sqrt(np.diag(pooled)), np.sqrt(np.diag(cov))
    assert np.allclose(cov / np.outer(sd_c, sd_c), pooled / np.outer(sd_p, sd_p))
    # With the sample variances on the diagonal it is the pooled covariance itself
    assert np.allclose(asset_regime_covariance(pooled, np.diag(pooled)), pooled)


def test_each_asset_gets_its_own_regime():
    """Asset 0 turns volatile at the end, asset 1 stays calm: only asset 0's
    high-variance state should be likely, and its forecast variance should
    be the larger one."""
    rng = np.random.default_rng(1)
    calm, wild = 0.005, 0.03
    x0 = np.concatenate([rng.normal(0, calm, 600), rng.normal(0, wild, 200), rng.normal(0, calm, 600), rng.normal(0, wild, 60)])
    x1 = np.concatenate([rng.normal(0, calm, 600), rng.normal(0, wild, 200), rng.normal(0, calm, 660)])
    X = np.column_stack([x0, x1])
    hmms = fit_asset_hmms(X, n_states=2)
    variances, high_var_prob = asset_regime_forecast(hmms, X)
    assert high_var_prob[0] > 0.9 and high_var_prob[1] < 0.1
    assert variances[0] > 10 * variances[1]


def test_walk_forward_option_adds_the_config_and_leaves_the_others_unchanged():
    ds, _ = make_synthetic_universe(n_days=500, seed=4)
    kwargs = dict(n_states=2, initial_window=252, constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30))
    base = run_walk_forward(ds.returns, **kwargs)
    with_option = run_walk_forward(ds.returns, asset_regime_states=2, **kwargs)

    assert set(with_option.weights) == set(base.weights) | {"erc_asset_regime"}
    for c in base.weights:
        assert np.array_equal(with_option.portfolio_returns[c], base.portfolio_returns[c])
        assert np.array_equal(with_option.weights[c], base.weights[c])
    w = with_option.weights["erc_asset_regime"]
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6) and (w >= -1e-6).all()
    assert np.isfinite(with_option.portfolio_returns["erc_asset_regime"]).all()
    assert len(with_option.asset_regime_history) == len(with_option.binding_constraints_history["erc_asset_regime"])
    assert base.asset_regime_history == []


def test_day_by_day_filter_matches_the_full_history_at_every_day():
    from asset_regime import AssetRegimeFilter, high_variance_probabilities

    ds, _ = make_synthetic_universe(n_days=700, seed=3)
    X = ds.returns.to_numpy()[:, :3]
    hmms = fit_asset_hmms(X[:400], n_states=2)
    filt = AssetRegimeFilter(hmms, X[:400])
    assert np.allclose(filt.high_variance_probability(), high_variance_probabilities(hmms, X[:400])[-1], atol=1e-12)
    for t in range(400, len(X)):
        filt.update(X[t])
        assert np.allclose(filt.high_variance_probability(), high_variance_probabilities(hmms, X[:t + 1])[-1], atol=1e-12)


_SMALL = dict(n_epochs=1, seq_len=20, cov_window=60, n_states=2)


def _lstm_available():
    from allocation_lstm import _CVXPYLAYERS_AVAILABLE, _TORCH_AVAILABLE
    return _TORCH_AVAILABLE and _CVXPYLAYERS_AVAILABLE


def test_asset_regime_variant_adds_one_input_per_asset_and_leaves_the_others_unchanged():
    import pytest
    if not _lstm_available():
        pytest.skip("requires torch and cvxpylayers")
    import torch
    from lstm_training import train_lstm_variants

    ds, _ = make_synthetic_universe(n_days=500, seed=0)
    plain, _ = train_lstm_variants(ds.returns, 400, seed=5, budget_floor=0.05, **_SMALL)
    models, _ = train_lstm_variants(ds.returns, 400, seed=5, budget_floor=0.05, asset_regime_states=2, **_SMALL)
    assert set(models) == {"baseline", "regime", "asset_regime"}
    assert models["asset_regime"].lstm.input_size == models["baseline"].lstm.input_size + ds.returns.shape[1]
    for v in ["baseline", "regime"]:
        for name, p in plain[v].state_dict().items():
            assert torch.equal(p, models[v].state_dict()[name])


def test_walk_forward_runs_the_asset_regime_lstm_and_leaves_the_others_unchanged():
    import pytest
    if not _lstm_available():
        pytest.skip("requires torch and cvxpylayers")
    from lstm_training import train_lstm_variants

    ds, _ = make_synthetic_universe(n_days=600, seed=7)
    models, fm = train_lstm_variants(ds.returns, 252, seed=1, budget_floor=0.05, asset_regime_states=2, **_SMALL)
    kwargs = dict(n_states=2, initial_window=252, lstm_feature_matrix=fm, lstm_seq_len=20, asset_regime_states=2,
                  constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30))
    pair = {v: models[v] for v in ["baseline", "regime"]}
    base = run_walk_forward(ds.returns, lstm_models=pair, **kwargs)
    full = run_walk_forward(ds.returns, lstm_models=models, **kwargs)

    assert set(full.weights) == set(base.weights) | {"lstm_asset_regime"}
    for c in base.weights:
        assert np.array_equal(full.portfolio_returns[c], base.portfolio_returns[c])
    w = full.weights["lstm_asset_regime"]
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6) and (w >= -1e-6).all()
    assert len(full.lstm_budgets_history["asset_regime"]) == len(full.binding_constraints_history["lstm_asset_regime"])

    with pytest.raises(ValueError):
        run_walk_forward(ds.returns, lstm_models=models, **{**kwargs, "asset_regime_states": None})
