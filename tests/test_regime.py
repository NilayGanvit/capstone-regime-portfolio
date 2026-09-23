import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe
from regime import GaussianHMM, bic_for_state_counts, bhattacharyya_distance, state_alignment_permutation


def _state_recovery_accuracy(pred_state, true_state):
    acc1 = (pred_state == true_state).mean()
    acc2 = (pred_state == 1 - true_state).mean()  # labels are arbitrary
    return max(acc1, acc2)


def test_hmm_converges():
    ds, _ = make_synthetic_universe(n_days=1000, seed=1)
    X = ds.returns.to_numpy()
    model = GaussianHMM(n_states=2, random_state=0).fit(X)
    assert model.converged_
    assert np.isfinite(model.log_likelihood_)


def test_hmm_recovers_synthetic_states():
    ds, true_state = make_synthetic_universe(n_days=1200, seed=2)
    X = ds.returns.to_numpy()
    true_state_aligned = true_state.iloc[1:].to_numpy()  # returns drops first row

    model = GaussianHMM(n_states=2, random_state=0).fit(X)
    filtered = model.filtered_probabilities(X)
    pred_state = filtered.argmax(axis=1)

    accuracy = _state_recovery_accuracy(pred_state, true_state_aligned)
    assert accuracy > 0.85, f"expected high state-recovery accuracy on synthetic data, got {accuracy}"


def test_filtered_probabilities_sum_to_one():
    ds, _ = make_synthetic_universe(n_days=500, seed=3)
    X = ds.returns.to_numpy()
    model = GaussianHMM(n_states=3, random_state=0).fit(X)
    filtered = model.filtered_probabilities(X)
    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-6)


def test_smoothed_differs_from_filtered_in_general():
    """Smoothed and filtered probabilities should NOT be identical in
    general -- if they were, the smoothing implementation likely isn't
    using future information at all, which would defeat the point of
    keeping them as distinct, separately-named outputs. Uses the same
    seed/length as test_hmm_recovers_synthetic_states, which is confirmed
    to produce genuine two-state switching rather than a degenerate fit
    that never really uses the second state (as a shorter/different seed
    can, especially with only 500 days -- state persistence is high, so
    short samples can fail to visit both states enough to matter)."""
    ds, _ = make_synthetic_universe(n_days=1200, seed=2)
    X = ds.returns.to_numpy()
    model = GaussianHMM(n_states=2, random_state=0).fit(X)
    filtered = model.filtered_probabilities(X)
    smoothed = model.smoothed_probabilities(X)
    assert not np.allclose(filtered, smoothed, atol=1e-3)


def test_bic_prefers_true_state_count_on_synthetic_data():
    ds, _ = make_synthetic_universe(n_days=1200, seed=2)
    X = ds.returns.to_numpy()
    results = bic_for_state_counts(X, candidates=(2, 3, 4), random_state=0)
    best_k = min(results, key=lambda k: results[k]["bic"])
    assert best_k == 2, f"expected BIC to favor the true 2-state generator, got {best_k}"


def test_state_alignment_identity_when_already_matched():
    """When new and reference states are already in the same order,
    permutation should be identity [0, 1]."""
    F = 5
    means = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0]])
    covars = np.array([np.eye(F), 2.0 * np.eye(F)])

    perm = state_alignment_permutation(means, covars, means, covars)
    assert np.array_equal(perm, [0, 1])


def test_state_alignment_recovers_swap():
    """When new states are swapped relative to reference, permutation
    should be [1, 0] to recover the reference order."""
    F = 5
    means = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0]])
    covars = np.array([np.eye(F), 2.0 * np.eye(F)])

    means_swapped = means[::-1]  # reverse
    covars_swapped = covars[::-1]

    perm = state_alignment_permutation(means_swapped, covars_swapped, means, covars)
    assert np.array_equal(perm, [1, 0])


def test_align_to_permutes_transmat_on_both_axes():
    """When align_to permutes states, it must permute transmat_ on both axes
    (rows and columns) to preserve transition semantics."""
    F = 3
    means = np.array([[0.0, 0.0, 0.0],
                      [2.0, 2.0, 2.0]])
    covars = np.array([np.eye(F), 2.0 * np.eye(F)])
    startprob = np.array([0.6, 0.4])
    transmat = np.array([[0.9, 0.1],
                         [0.2, 0.8]])

    # Build reference HMM with states in one order
    ref_hmm = GaussianHMM(n_states=2)
    ref_hmm.means_ = means.copy()
    ref_hmm.covars_ = covars.copy()
    ref_hmm.startprob_ = startprob.copy()
    ref_hmm.transmat_ = transmat.copy()
    ref_hmm.n_features_ = F

    # Build new HMM with swapped state order
    new_hmm = GaussianHMM(n_states=2)
    new_hmm.means_ = means[::-1].copy()
    new_hmm.covars_ = covars[::-1].copy()
    new_hmm.startprob_ = startprob[::-1].copy()
    # Permute transmat on both axes: if we swap states 0<->1, then
    # transmat[0,0] -> transmat[1,1], etc.
    new_hmm.transmat_ = transmat[::-1][:, ::-1].copy()
    new_hmm.n_features_ = F

    # Align new to reference
    new_hmm.align_to(ref_hmm)

    # After alignment, new_hmm should match ref_hmm
    assert np.allclose(new_hmm.means_, ref_hmm.means_)
    assert np.allclose(new_hmm.covars_, ref_hmm.covars_)
    assert np.allclose(new_hmm.startprob_, ref_hmm.startprob_)
    assert np.allclose(new_hmm.transmat_, ref_hmm.transmat_)


def test_fit_with_init_from_and_zero_iterations_keeps_reference_params():
    """init_from should be copied in as the EM starting point; with
    n_iter=0 the EM loop body never executes, so the result must equal
    the reference's params exactly (as copies, not the same object) --
    this locks in the warm-start wiring deterministically, independent of
    EM's iterative behavior."""
    ds, _ = make_synthetic_universe(n_days=500, seed=4)
    X = ds.returns.to_numpy()
    reference = GaussianHMM(n_states=3, random_state=0).fit(X)

    warm = GaussianHMM(n_states=3, n_iter=0).fit(X, init_from=reference)

    assert np.array_equal(warm.means_, reference.means_)
    assert np.array_equal(warm.covars_, reference.covars_)
    assert np.array_equal(warm.transmat_, reference.transmat_)
    assert np.array_equal(warm.startprob_, reference.startprob_)
    assert warm.means_ is not reference.means_
    assert warm.covars_ is not reference.covars_
    assert warm.transmat_ is not reference.transmat_
    assert warm.startprob_ is not reference.startprob_


def test_fit_init_from_rejects_mismatched_n_states():
    ds, _ = make_synthetic_universe(n_days=500, seed=4)
    X = ds.returns.to_numpy()
    reference = GaussianHMM(n_states=3, random_state=0).fit(X)

    try:
        GaussianHMM(n_states=2, n_iter=0).fit(X, init_from=reference)
        assert False, "expected ValueError on n_states mismatch"
    except ValueError:
        pass
