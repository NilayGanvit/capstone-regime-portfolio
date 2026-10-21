"""
Audit of the ERC risk-budgeting solver and the constraint projection on the
real ten-ETF walk-forward.

Two separate questions:

  1. Solver. allocation_erc.solve_risk_budget minimises the squared error between
     risk-contribution shares and equal budgets with SLSQP. That objective is
     non-convex in the weights, so SLSQP can stop at a local optimum. The exact
     answer is available from the strictly convex log-barrier problem
     (allocation_erc.solve_risk_budget_convex). On every decision date this
     compares the two raw targets: how many assets sit at zero weight, and the
     worst risk-share deviation from 10%.
  2. Projection. constraints.project_onto_constraints moves the raw target to the
     nearest point (Euclidean) that satisfies the weight cap and the turnover
     limit; it does not minimise risk-contribution deviation. On each date this
     compares three weight vectors against the same drifted holdings: the
     frozen pipeline's final weights, the exact ERC target, and the exact target
     passed through the same projection. If the exact target already satisfies
     the cap and turnover limit, the projection leaves it unchanged, and any
     post-projection deviation in the frozen pipeline comes from the raw
     target, not from the projection.

The solver inputs are captured from an actual ERC-only walk-forward (the same
call run_real_data.py makes, without LSTM models), so nothing is reconstructed.
Weights only: no returns or performance are computed here.

Usage:
    python scripts/verification/audit_erc_solver.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import walkforward as wf  # noqa: E402
from allocation_erc import (  # noqa: E402
    regularize_covariance, risk_contribution_shares, solve_risk_budget_convex,
)
from constraints import ConstraintSpec, project_onto_constraints, turnover  # noqa: E402
from data import M2Calendar, PriceDataset, initial_window_length, stage_labels  # noqa: E402
from run_real_data import run_primary_walk_forward  # noqa: E402

SPEC = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30)
ZERO = 1e-4  # a weight below this counts as pinned at zero
OUT = ROOT / "outputs" / "erc_solver_audit.csv"
FIXTURE = ROOT / "tests" / "fixtures" / "erc_solver_cases.npz"
FLAGGED = ["2020-03-31", "2022-04-29", "2026-01-30"]  # dates reported by the independent audit


def best_feasible_risk_parity(cov, w_drift, starts):
    """Smallest achievable squared risk-share error within the same constraints
    (long-only, fully invested, weight cap, gross turnover limit from the drifted
    holdings), found by multistart SLSQP. The objective is non-convex, so this is
    an upper bound on the true optimum's error, not a certificate of it."""
    n = len(w_drift)

    def sse(x):
        return float(np.sum((risk_contribution_shares(x[:n], cov) - 1.0 / n) ** 2))

    cons = [
        {"type": "eq", "fun": lambda x: x[:n].sum() - 1.0},
        {"type": "eq", "fun": lambda x: x[:n] - w_drift - x[n:2 * n] + x[2 * n:]},
        {"type": "ineq", "fun": lambda x: SPEC.max_turnover - x[n:].sum()},
    ]
    bounds = [(SPEC.lower, SPEC.upper)] * n + [(0, None)] * (2 * n)
    best = None
    for w0 in starts:
        d = w0 - w_drift
        x0 = np.concatenate([w0, np.clip(d, 0, None), np.clip(-d, 0, None)])
        res = minimize(sse, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        w = np.clip(res.x[:n], 0, None)
        w = w / w.sum()
        if turnover(w, w_drift) <= SPEC.max_turnover + 1e-6 and w.max() <= SPEC.upper + 1e-6 and (best is None or sse(np.concatenate([w, [0] * 2 * n])) < best[0]):
            best = (sse(np.concatenate([w, [0] * 2 * n])), w)
    return best[1]


def main() -> None:
    solves, projections = [], []
    orig_base, orig_reg, orig_proj = wf.erc_baseline_weights, wf.erc_regime_weights, wf.project_onto_constraints

    def base(pooled_cov, shrinkage=0.10):
        w = orig_base(pooled_cov, shrinkage)
        solves.append(("erc_baseline", regularize_covariance(pooled_cov, shrinkage), w))
        return w

    def reg(state_covs, probs, shrinkage=0.10):
        w = orig_reg(state_covs, probs, shrinkage)
        solves.append(("erc_regime", regularize_covariance(np.tensordot(probs, state_covs, axes=(0, 0)), shrinkage), w))
        return w

    def proj(w_raw, w_drift, spec):
        w = orig_proj(w_raw, w_drift, spec)
        projections.append((np.array(w_raw), np.array(w_drift), np.array(w)))
        return w

    wf.erc_baseline_weights, wf.erc_regime_weights, wf.project_onto_constraints = base, reg, proj
    try:
        ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
        calendar = M2Calendar()
        result = run_primary_walk_forward(ds, initial_window_length(ds.returns.index, calendar))
    finally:
        wf.erc_baseline_weights, wf.erc_regime_weights, wf.project_onto_constraints = orig_base, orig_reg, orig_proj

    configs = list(result.weights)
    n_rebal = len(result.binding_constraints_history["erc_baseline"])
    assert len(solves) == 2 * n_rebal and len(projections) == len(configs) * n_rebal, (len(solves), len(projections), n_rebal)
    dates = [pd.Timestamp(e["date"]) for e in result.binding_constraints_history["erc_baseline"]]
    final = (stage_labels(pd.DatetimeIndex(dates), calendar) == "final_test").to_numpy()

    rows, fixture = [], {}
    for r, date in enumerate(dates):
        if not final[r]:
            continue
        for k, config in enumerate(["erc_baseline", "erc_regime"]):
            name, cov, w_slsqp = solves[2 * r + k]
            assert name == config
            w_raw, w_drift, w_final = projections[r * len(configs) + configs.index(config)]
            assert np.allclose(w_raw, w_slsqp), "captured projection input is not the captured solver output"
            w_exact = solve_risk_budget_convex(cov)
            w_exact_proj = project_onto_constraints(w_exact, w_drift, SPEC)
            dev = lambda w: float(np.abs(risk_contribution_shares(w, cov) - 0.1).max())  # noqa: E731
            rows.append({
                "date": date.date(), "config": config,
                "slsqp_assets_at_zero": int((w_slsqp < ZERO).sum()),
                "slsqp_max_share_dev": dev(w_slsqp),
                "exact_max_share_dev": dev(w_exact),
                "exact_max_weight": float(w_exact.max()),
                "exact_target_turnover": float(turnover(w_exact, w_drift)),
                "exact_target_feasible": bool(w_exact.max() <= SPEC.upper + 1e-9 and turnover(w_exact, w_drift) <= SPEC.max_turnover + 1e-9),
                "frozen_final_turnover": float(turnover(w_final, w_drift)),
                "frozen_final_max_share_dev": dev(w_final),
                "exact_projected_turnover": float(turnover(w_exact_proj, w_drift)),
                "exact_projected_max_share_dev": dev(w_exact_proj),
            })
            if turnover(w_final, w_drift) >= SPEC.max_turnover - 1e-6:
                rng = np.random.default_rng(0)
                starts = [w_exact_proj, w_final, w_drift] + [rng.dirichlet(np.ones(len(w_drift)) * 3) for _ in range(6)]
                w_best = best_feasible_risk_parity(cov, w_drift, starts)
                rows[-1].update({"best_feasible_max_share_dev": dev(w_best), "best_feasible_turnover": float(turnover(w_best, w_drift))})
                fixture[f"cov_{config}_{date:%Y%m%d}"] = cov
                fixture[f"w_drift_{config}_{date:%Y%m%d}"] = w_drift
            if config == "erc_regime" and date.strftime("%Y-%m-%d") in FLAGGED:
                fixture[f"cov_{date:%Y%m%d}"] = cov
                fixture[f"w_slsqp_{date:%Y%m%d}"] = w_slsqp
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False, float_format="%.6g")
    FIXTURE.parent.mkdir(exist_ok=True)
    np.savez_compressed(FIXTURE, **fixture)

    pd.set_option("display.width", 200)
    for config, g in table.groupby("config"):
        zero = g[g.slsqp_assets_at_zero > 0]
        print(f"\n=== {config}: {len(g)} final-period decision dates ===")
        print(f"SLSQP raw target with at least one asset at zero weight: {len(zero)} dates "
              f"({', '.join(map(str, zero.date))})" if len(zero) else "SLSQP raw target with an asset at zero weight: none")
        print(f"max risk-share deviation, SLSQP raw: worst {g.slsqp_max_share_dev.max():.4f}, median {g.slsqp_max_share_dev.median():.2e}; "
              f"exact convex solution: worst {g.exact_max_share_dev.max():.1e}")
        print(f"exact ERC target violates the {SPEC.upper:.0%} cap or the {SPEC.max_turnover:.0%} turnover limit on "
              f"{int((~g.exact_target_feasible).sum())} dates")
        bind = g[g.frozen_final_turnover >= SPEC.max_turnover - 1e-6]
        print(f"frozen pipeline turnover-binding dates: {len(bind)}")
    flag = table[table.date.astype(str).isin(FLAGGED) & (table.config == "erc_regime")]
    print("\nDates reported by the independent audit (erc_regime):")
    print(flag[["date", "slsqp_assets_at_zero", "slsqp_max_share_dev", "exact_max_share_dev", "exact_target_turnover",
                "frozen_final_turnover", "frozen_final_max_share_dev", "exact_projected_turnover", "exact_projected_max_share_dev"]].round(4).to_string(index=False))
    b = table[(table.config == "erc_regime") & (table.frozen_final_turnover >= SPEC.max_turnover - 1e-6)]
    print("\nTurnover-binding dates in the frozen pipeline (erc_regime): max risk-share deviation from 10%")
    print(b[["date", "slsqp_assets_at_zero", "frozen_final_max_share_dev", "exact_target_turnover", "exact_target_feasible",
             "exact_projected_max_share_dev", "best_feasible_max_share_dev", "best_feasible_turnover"]].round(4).to_string(index=False))
    print(f"\nSaved {OUT} and {FIXTURE}")


if __name__ == "__main__":
    main()
