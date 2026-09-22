import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reliability import ReliabilityTracker, blend_weights, select_forgetting_factor


def test_pi_rises_when_m1_consistently_wins():
    tracker = ReliabilityTracker(alpha=0.97, pi0=0.5)
    for _ in range(200):
        tracker.update(log_l0=-5.0, log_l1=-2.0)
    assert tracker.pi_ > 0.9


def test_pi_falls_when_m0_consistently_wins():
    tracker = ReliabilityTracker(alpha=0.97, pi0=0.5)
    for _ in range(200):
        tracker.update(log_l0=-2.0, log_l1=-5.0)
    assert tracker.pi_ < 0.1


def test_pi_reverts_toward_half_when_tied():
    tracker = ReliabilityTracker(alpha=0.97, pi0=0.9)
    for _ in range(300):
        tracker.update(log_l0=-2.0, log_l1=-2.0)
    assert abs(tracker.pi_ - 0.5) < 0.05


def test_pi_never_reaches_hard_zero_or_one():
    tracker = ReliabilityTracker(alpha=0.99, pi0=0.5)
    for _ in range(5000):
        tracker.update(log_l0=-50.0, log_l1=-1.0)
    assert 0.0 < tracker.pi_ < 1.0


def test_blend_weights_is_linear_interpolation():
    w_regime = np.array([0.6, 0.4])
    w_baseline = np.array([0.3, 0.7])
    for pi in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = blend_weights(w_regime, w_baseline, pi)
        expected = pi * w_regime + (1 - pi) * w_baseline
        assert np.allclose(blended, expected)


def test_select_forgetting_factor_returns_a_candidate():
    log_l0 = np.full(100, -2.0)
    log_l1 = np.full(100, -5.0)
    alpha = select_forgetting_factor(log_l0, log_l1, candidates=(0.9, 0.95, 1.0))
    assert alpha in (0.9, 0.95, 1.0)
