"""
Hierarchical Risk Parity (HRP) allocator.

Corresponds to the M2 scope note deferring HRP to a secondary robustness
check ("out of scope for this pass ... secondary robustness checks only if
time allows" -- see README "Known scope limitations"). Implements Lopez de
Prado's (2016) tree-based construction as an alternative to
allocation_erc's ERC risk parity, used here only to test whether ERC's
regime effect (M2 research question 1) survives under a completely
different risk-parity construction -- not as a third primary allocator
alongside ERC/LSTM. Wired into walkforward.run_walk_forward as an opt-in
`include_hrp` toggle (default False), consumed by
scripts/run_hrp_and_nu_robustness.py, not by the primary evaluation run.

    correlation -> distance matrix -> single-linkage hierarchical
    clustering -> quasi-diagonal leaf order -> recursive bisection

Unlike allocation_erc.solve_risk_budget (an SLSQP solve matching risk
contributions to explicit target budgets), HRP never inverts the full
covariance matrix, and produces long-only, fully-invested weights by
construction (inverse-variance cluster weights and a convex split factor
at every bisection) -- no separate projection/clipping step is needed to
guarantee those two properties, though the shared 0.60 per-asset cap and
turnover limit still apply downstream, exactly as for ERC, via
constraints.project_onto_constraints at evaluation time.

Owner: AnnaLisa & Nilay (traditional allocation, same family as
allocation_erc.py).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from allocation_erc import regularize_covariance


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(0.5*(1-corr_ij)) -- a proper metric (Lopez de Prado,
    2016) turning correlation into a distance hierarchical clustering can
    consume. Clipped to [0, 1] before the sqrt to guard against tiny
    negative values from floating-point round-off on corr_ij == 1."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def quasi_diagonal_order(cov: np.ndarray) -> list[int]:
    """Single-linkage hierarchical clustering on the correlation-distance
    matrix, then the dendrogram's leaf order. This *is* quasi-
    diagonalization (reordering assets so the most similar ones sit next
    to each other) -- no separate recursive step is needed, since
    scipy's `to_tree(...).pre_order()` already returns leaves in that
    order."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = correlation_distance(corr)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    return to_tree(link, rd=False).pre_order()


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Variance of the inverse-variance-weighted portfolio restricted to
    `indices` -- the split criterion recursive_bisection allocates against
    at every node, per Lopez de Prado (2016)."""
    sub_cov = cov[np.ix_(indices, indices)]
    inv_var = 1.0 / np.diag(sub_cov)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def recursive_bisection(cov: np.ndarray, sorted_indices: list[int]) -> np.ndarray:
    """Allocate weights top-down over the quasi-diagonal order: at each
    node, split the cluster in two contiguous halves, weight each half by
    1 - var_half / (var_left + var_right) (the lower-variance half gets
    more weight), and recurse until every cluster is a single asset.
    Returns weights indexed by the *original* asset order (0..n-1), not
    by `sorted_indices`.
    """
    n = cov.shape[0]
    weights = np.ones(n)
    clusters = [list(sorted_indices)]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters
    return weights / weights.sum()


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Full HRP construction on an already-regularized covariance:
    quasi-diagonal order, then recursive bisection."""
    order = quasi_diagonal_order(cov)
    return recursive_bisection(cov, order)


def hrp_baseline_weights(pooled_cov: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    """HRP against a single pooled covariance estimate, no regime
    separation -- the HRP counterpart to allocation_erc.erc_baseline_weights,
    same regularization rule."""
    cov = regularize_covariance(pooled_cov, shrinkage)
    return hrp_weights(cov)


def hrp_regime_weights(
    state_covs: np.ndarray,
    regime_probs: np.ndarray,
    shrinkage: float = 0.10,
) -> np.ndarray:
    """HRP against the same regime-probability-weighted covariance mixture
    erc_regime_weights uses -- the HRP counterpart to
    allocation_erc.erc_regime_weights.
        Sigma_t = sum_k P(S_t+1 = k | F_t) * Sigma_k
    then regularized identically to the baseline.
    """
    mixture_cov = np.tensordot(regime_probs, state_covs, axes=(0, 0))
    cov = regularize_covariance(mixture_cov, shrinkage)
    return hrp_weights(cov)
