"""
Entry point: python scripts/run_walk_forward.py

Wires together data loading, regime inference, allocation, and
evaluation into a single run of the walk-forward test described in
docs/problem_statement.md.
"""

from capstone.data.universe import load_universe
from capstone.evaluation.walk_forward import run_walk_forward


def main():
    data = load_universe()
    config = {
        # TODO: development/test split dates, HMM state count,
        # LSTM hyperparameters, constraint bounds, cost model, etc.
    }
    results = run_walk_forward(data, config)
    print(results)


if __name__ == "__main__":
    main()
