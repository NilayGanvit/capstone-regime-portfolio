import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe
from regime import GaussianHMM, bic_for_state_counts


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
