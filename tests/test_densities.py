import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe
from regime import GaussianHMM
from densities import M0PooledStudentT, M1RegimeMixtureStudentT, fit_shared_nu, weighted_mean_cov


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
