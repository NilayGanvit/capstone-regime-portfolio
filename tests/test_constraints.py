import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constraints import ConstraintSpec, drift_weights, project_onto_constraints, turnover, binding_constraints


def test_drift_weights_sums_to_one():
    w_prev = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    gross_ret = np.array([1.05, 0.98, 1.10, 1.00, 0.95])
    w_drift = drift_weights(w_prev, gross_ret)
    assert np.isclose(w_drift.sum(), 1.0)


def test_projection_respects_bounds_and_budget():
    w_prev = np.full(5, 0.2)
    w_drift = drift_weights(w_prev, np.array([1.05, 0.98, 1.10, 1.00, 0.95]))
    spec = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.50)
    w_raw = np.array([0.30, 0.05, 0.30, 0.05, 0.30])
    w_final = project_onto_constraints(w_raw, w_drift, spec)
    assert np.isclose(w_final.sum(), 1.0, atol=1e-6)
    assert (w_final >= spec.lower - 1e-6).all()
    assert (w_final <= spec.upper + 1e-6).all()


def test_projection_respects_tight_turnover_limit():
    w_prev = np.full(5, 0.2)
    w_drift = drift_weights(w_prev, np.array([1.05, 0.98, 1.10, 1.00, 0.95]))
    spec = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.05)
    w_raw = np.array([0.30, 0.05, 0.30, 0.05, 0.30])
    w_final = project_onto_constraints(w_raw, w_drift, spec)
    assert turnover(w_final, w_drift) <= spec.max_turnover + 1e-3


def test_binding_constraints_flags_turnover_when_limited():
    w_prev = np.full(5, 0.2)
    w_drift = drift_weights(w_prev, np.array([1.05, 0.98, 1.10, 1.00, 0.95]))
    spec = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.05)
    w_raw = np.array([0.30, 0.05, 0.30, 0.05, 0.30])
    w_final = project_onto_constraints(w_raw, w_drift, spec)
    flags = binding_constraints(w_final, w_drift, spec)
    assert flags["turnover_binding"] is True


def test_unconstrained_case_matches_raw_target_when_feasible():
    w_prev = np.full(5, 0.2)
    w_drift = drift_weights(w_prev, np.array([1.01, 0.99, 1.02, 0.98, 1.00]))
    spec = ConstraintSpec(lower=0.0, upper=1.0, max_turnover=1.0)  # loose
    w_raw = np.array([0.30, 0.10, 0.30, 0.10, 0.20])
    w_final = project_onto_constraints(w_raw, w_drift, spec)
    assert np.allclose(w_final, w_raw, atol=1e-3)
