"""
Selection against 1/N across validation folds, as pre-registered in
docs/verification/crisis_folds_selection_rule_preregistration.md.

    per fold: Sharpe(candidate) - Sharpe(1/N), with a bootstrap interval
      -> out if any fold's interval lies entirely below 0
      -> among the rest, the highest worst-fold difference; ties at three
         decimals go to the simpler candidate (earlier in `order`)
"""

from __future__ import annotations

import pandas as pd

BENCHMARK = "equal_weight"


def benchmark_relative_choice(table: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, str]:
    """`table` has one row per (candidate, fold) with columns candidate, fold, delta, ci_low,
    ci_high, where delta is the candidate's Sharpe minus 1/N's; the benchmark needs no rows.
    Returns one row per candidate in `order` (worst-fold delta, eligibility) and the choice."""
    rows = [{"candidate": BENCHMARK, "worst_fold_delta": 0.0, "worst_fold": None, "eligible": True}]
    for c in order:
        if c == BENCHMARK:
            continue
        t = table[table.candidate == c]
        worst = t.loc[t.delta.idxmin()]
        rows.append({"candidate": c, "worst_fold_delta": float(worst.delta), "worst_fold": worst.fold,
                     "eligible": bool((t.ci_high >= 0).all())})
    summary = pd.DataFrame(rows).set_index("candidate").loc[[BENCHMARK] + [c for c in order if c != BENCHMARK]]
    eligible = summary[summary.eligible]
    best = eligible.worst_fold_delta.round(3).max()
    choice = next(c for c in eligible.index if round(eligible.loc[c, "worst_fold_delta"], 3) == best)
    summary["chosen"] = summary.index == choice
    return summary.reset_index(), choice
