import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_hrp import (
    quasi_diagonal_order, recursive_bisection, hrp_weights,
    hrp_baseline_weights, hrp_regime_weights,
)


def _two_cluster_cov(n_per_cluster=3, within_corr=0.9, across_corr=0.0, seed=0):
    """Two tight clusters of `n_per_cluster` assets each, highly correlated
    within a cluster and uncorrelated across clusters -- a covariance where
    quasi-diagonalization has an unambiguous right answer (the two clusters
    should end up contiguous in the returned order)."""
    n = 2 * n_per_cluster
    corr = np.full((n, n), across_corr)
    corr[:n_per_cluster, :n_per_cluster] = within_corr
    corr[n_per_cluster:, n_per_cluster:] = within_corr
    np.fill_diagonal(corr, 1.0)
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.08, 0.22, n)
    return np.outer(vols, vols) * corr


def test_quasi_diagonal_order_groups_clusters_contiguously():
    n_per_cluster = 3
    cov = _two_cluster_cov(n_per_cluster)
    order = quasi_diagonal_order(cov)
    assert sorted(order) == list(range(2 * n_per_cluster))

    cluster_of = lambda i: 0 if i < n_per_cluster else 1
    labels = [cluster_of(i) for i in order]
    # Each cluster's members should occupy one contiguous run in the order.
    assert labels == sorted(labels) or labels == sorted(labels, reverse=True)


def test_recursive_bisection_weights_sum_to_one_and_long_only():
    cov = _two_cluster_cov()
    order = quasi_diagonal_order(cov)
    w = recursive_bisection(cov, order)
    assert np.isclose(w.sum(), 1.0, atol=1e-10)
    assert (w > 0).all()


def test_hrp_weights_are_long_only_fully_invested():
    cov = _two_cluster_cov(n_per_cluster=4, seed=1)
    w = hrp_weights(cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    assert (w > 0).all()


def test_hrp_underweights_the_more_volatile_cluster():
    """The low-volatility cluster should end up with more aggregate weight
    than the high-volatility one -- HRP's inverse-variance split at every
    node should propagate that preference to the top level."""
    n_per_cluster = 3
    cov = _two_cluster_cov(n_per_cluster, seed=2)
    cov[n_per_cluster:, n_per_cluster:] *= 9.0  # second cluster much more volatile
    cov[n_per_cluster:, :n_per_cluster] *= 3.0
    cov[:n_per_cluster, n_per_cluster:] *= 3.0
    w = hrp_weights(cov)
    assert w[:n_per_cluster].sum() > w[n_per_cluster:].sum()


def test_hrp_baseline_weights_matches_direct_call_on_regularized_cov():
    cov = _two_cluster_cov(seed=3)
    shrinkage = 0.2
    from allocation_erc import regularize_covariance
    expected = hrp_weights(regularize_covariance(cov, shrinkage))
    actual = hrp_baseline_weights(cov, shrinkage)
    assert np.allclose(expected, actual)


def test_hrp_regime_weights_differ_across_very_different_state_covariances():
    n = 6
    calm = np.eye(n) * 0.0004
    stress = _two_cluster_cov(n // 2, seed=4) * 5
    state_covs = np.array([calm, stress])
    w_calm_dominant = hrp_regime_weights(state_covs, regime_probs=np.array([0.99, 0.01]))
    w_stress_dominant = hrp_regime_weights(state_covs, regime_probs=np.array([0.01, 0.99]))
    assert not np.allclose(w_calm_dominant, w_stress_dominant, atol=1e-3)
