import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe
from regime import GaussianHMM
from densities import (
    M0PooledStudentT, M1RegimeMixtureStudentT, fit_shared_nu, fit_per_regime_nu, weighted_mean_cov,
)


def test_weighted_mean_cov_matches_equal_weights_case():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    equal_w = np.ones(500)
    mean, cov = weighted_mean_cov(X, equal_w)
    assert np.allclose(mean, X.mean(axis=0), atol=1e-8)
    assert np.allclose(cov, np.cov(X.T, bias=True), atol=1e-6)


def test_fit_shared_nu_returns_reasonable_value():
    rng = np.random.default_rng(0)
    # Student-t(df=5) samples, standardized
    x = rng.standard_t(df=5, size=20000)
    nu = fit_shared_nu(x)
    assert 2.5 <= nu <= 15, f"expected nu near the true df=5, got {nu}"


def test_m1_outpredicts_m0_on_regime_switching_data():
    ds, _ = make_synthetic_universe(n_days=1200, seed=2)
    X = ds.returns.to_numpy()
    train, test = X[:800], X[800:850]

    hmm = GaussianHMM(n_states=2, random_state=0).fit(train)
    filtered_train = hmm.filtered_probabilities(train)

    std_resid = ((train - train.mean(axis=0)) / train.std(axis=0)).ravel()
    nu = fit_shared_nu(std_resid)
    m0 = M0PooledStudentT(nu=nu).fit(train)
    m1 = M1RegimeMixtureStudentT(nu=nu).fit(train, filtered_train)

    running = np.vstack([train, test])
    ll0, ll1 = 0.0, 0.0
    for i in range(len(test)):
        cutoff = 800 + i
        hist = running[:cutoff]
        filtered = hmm.filtered_probabilities(hist)[-1]
        predicted = filtered @ hmm.transmat_
        r_next = running[cutoff]
        ll0 += m0.log_density(r_next)
        ll1 += m1.log_density(r_next, predicted)

    assert ll1 > ll0, "M1 (regime-aware) should out-predict M0 (pooled) when data genuinely has regimes"


def test_fit_per_regime_nu_returns_one_value_per_state():
    rng = np.random.default_rng(3)
    T, n_features, n_states = 600, 3, 2
    X = rng.standard_t(df=5, size=(T, n_features)) * 0.01
    # Deterministic hard assignment (not soft filtered probs) is fine here
    # -- fit_per_regime_nu only needs a (T, n_states) weight matrix.
    probs = np.zeros((T, n_states))
    probs[: T // 2, 0] = 1.0
    probs[T // 2 :, 1] = 1.0
    nus = fit_per_regime_nu(X, probs)
    assert nus.shape == (n_states,)
    assert (nus > 2).all()


def test_m1_accepts_scalar_or_per_regime_nu_array():
    """A per-state nu array should change the density from the
    shared-scalar case whenever the states actually differ in tail
    behavior -- otherwise per_regime_nu would be a silent no-op."""
    rng = np.random.default_rng(4)
    T, n_features, n_states = 400, 3, 2
    X = rng.normal(size=(T, n_features)) * 0.01
    probs = rng.dirichlet(np.ones(n_states), size=T)

    nu_shared = 8.0
    nu_per_regime = np.array([4.0, 20.0])  # deliberately very different

    m1_shared = M1RegimeMixtureStudentT(nu=nu_shared).fit(X, probs)
    m1_per_regime = M1RegimeMixtureStudentT(nu=nu_per_regime).fit(X, probs)

    r = X[0]
    predicted = np.array([0.5, 0.5])
    ld_shared = m1_shared.log_density(r, predicted)
    ld_per_regime = m1_per_regime.log_density(r, predicted)
    assert np.isfinite(ld_shared) and np.isfinite(ld_per_regime)
    assert not np.isclose(ld_shared, ld_per_regime)
