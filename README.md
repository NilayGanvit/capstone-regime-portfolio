# Reliability-Gated Regime-Aware Dynamic Portfolio Allocation

MScFE 690 Capstone Project — Track 7 (Medium/Long-Term Trading Models)

Silvio Mazzaro · AnnaLisa Aaron · Nilay Jayantibhai Ganvit

## Problem statement (summary)

Can information about market regimes improve medium-term portfolio allocation?
We compare a long-only Equal Risk Contribution (ERC) Risk Parity allocator
against an LSTM-based dynamic risk-budgeting allocator, each in a baseline
(no regime input) and regime-aware (HMM state probabilities as input)
version, plus a reliability-gated blend of the two driven by a recursively
updated model-confidence weight `pi_t`. See `docs/problem_statement.md`
for the full text from the Module 2 submission.

Research questions:
1. Is regime information explicitly helpful for net risk-adjusted
   performance versus the same allocator without regime information?
2. Does the LSTM-based dynamic risk-budgeting allocator benefit more
   from regime information than fixed-budget ERC Risk Parity?
3. Is there value added by reliability-based blending beyond the raw
   regime probabilities?

## Repository structure

```
src/capstone/
    data/           universe loading, point-in-time data timing
    regime/         3-state Gaussian HMM, M0/M1 reliability layer
    allocation/     ERC risk parity, LSTM dynamic risk budgeting
    optimization/   shared risk-budgeting solver, constraints/projection
    evaluation/     performance metrics, walk-forward harness
tests/              unit tests (mirror src/capstone structure)
scripts/            entry points (e.g. run_walk_forward.py)
docs/               problem statement, notes, diagrams
```

## Module ownership

| Module | Owner |
|---|---|
| `data/` | shared |
| `regime/` (HMM + reliability) | Silvio |
| `allocation/` (ERC + LSTM) | AnnaLisa / Nilay |
| `optimization/` (risk budgeting + constraints) | Nilay |
| `evaluation/` (metrics + walk-forward harness) | Nilay |

Every function currently raises `NotImplementedError` — these are
scaffolds with signatures and docstrings matching the Module 2
pseudocode, not working implementations. Fill in your module and open
a PR against `main`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Non-anticipative discipline

`evaluation/walk_forward.py` is the single place that ties every module
together into the actual out-of-sample test. Any information leakage
(smoothed HMM states, full-sample fits, same-day predictive densities,
turnover computed on raw budgets instead of realized weights) invalidates
the whole result. Review changes to this file, and to any function it
calls, more carefully than anything else in the repo.

## Status

Module 3 (Literature Review & Competitor Analysis) — scaffolding stage.
Each module owner should aim to get their piece producing real,
if preliminary, output on a short window before the next submission.
