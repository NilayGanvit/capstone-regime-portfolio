"""
Walk-forward harness.

Owner: Nilay (integrates every other module)

This is the direct implementation of the M2 pseudocode:

    01 Initialization  (before the test period)
       - set universe, monthly calendar, costs, common constraints
       - define development period, test windows, validation rules
       - fit initial HMM, M0/M1, and paired LSTM risk-budgeting models
       - set pi = 0.5; save one-step-ahead predictive densities

    02 Daily walk-forward  (one chronological loop)
       for each trading day t:
         - at open: project scheduled targets onto constraints, execute
         - accrue held-portfolio return through the close
         - at close: observe r(t), score under M0/M1 saved at t-1
         - update pi from prior weights + log-likelihoods
         - if a scheduled refit is due: refit using F(t) only,
           tune only on earlier chronological validation windows,
           preserve past forecasts/pi/portfolio history
         - filter HMM states, predict probabilities for t+1
         - save M0/M1 predictive densities for r(t+1)
         - if t is a month-end allocation date:
             compute baseline + regime-aware ERC and LSTM targets,
             blend each matched pair using current pi,
             schedule all six configurations + equal-weight benchmark

    03 Evaluation  (same test dates)
       - compare paired net performance across all seven configurations
       - report uncertainty, model diagnostics, and ablation results

Every step here MUST only use information available as of its own
timestamp -- this module is the single place where a leakage bug
would silently invalidate the whole out-of-sample test, so treat
edits here with the most scrutiny of anything in the repo.
"""

from __future__ import annotations


def run_walk_forward(universe_data, config: dict) -> dict:
    """
    TODO: orchestrate the full loop above, calling into:
      - capstone.regime.hmm / capstone.regime.reliability
      - capstone.allocation.erc / capstone.allocation.lstm_risk_budget
      - capstone.optimization.risk_budgeting / capstone.optimization.constraints
      - capstone.evaluation.metrics

    Returns a dict of net return series, one per configuration
    (ERC x {baseline, regime, blend}, LSTM x {baseline, regime, blend},
    equal-weight benchmark), plus diagnostics (pi path, state
    frequencies, predictive log scores, binding constraints).
    """
    raise NotImplementedError
