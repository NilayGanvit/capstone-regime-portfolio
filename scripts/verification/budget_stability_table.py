"""
Recompute Table 4 (risk-budget stability across ten LSTM seeds, ordinary versus bounded Softmax)
from the saved budgets, and check the result against the committed summary files. Writes nothing.

The saved budgets are outputs/bounded_softmax_u050_200epoch_seed_stability_diagnostic.csv: the
LSTM risk budgets at 80 monthly decision dates for seeds 0-9, baseline and regime networks,
bounded Softmax with a 5% floor, 200 epochs, trained on data through 2014. This script does not
train anything. It is the summary step of the Colab notebook that produced those files, moved
here with repository-relative paths and without overwriting them; the training step is not
reproduced here.

The ordinary-Softmax reference values are the four-decimal figures the notebook used; they are
checked against outputs/lstm_seed_stability_summary.csv.

Usage:
    python scripts/verification/budget_stability_table.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "outputs"
PREFIX = "bounded_softmax_u050_200epoch"
ASSETS = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"]
BUDGET_COLS = [f"budget_{a}" for a in ASSETS]
SEEDS, VARIANTS, FLOOR = list(range(10)), ["baseline", "regime"], 0.05

# Ordinary-Softmax, 10 seeds, 200 epochs (outputs/lstm_seed_stability_summary.csv, rounded)
ORDINARY = pd.DataFrame([
    {"variant": "baseline", "ordinary_mean_pairwise_L1": 0.9036, "ordinary_median_pairwise_L1": 0.8284,
     "ordinary_max_pairwise_L1": 1.9541, "ordinary_mean_HHI": 0.5040, "ordinary_mean_cross_seed_HHI_sd": 0.1548,
     "ordinary_mean_asset_cross_seed_sd": 0.0870, "ordinary_max_asset_cross_seed_sd": 0.2168},
    {"variant": "regime", "ordinary_mean_pairwise_L1": 0.8780, "ordinary_median_pairwise_L1": 0.8324,
     "ordinary_max_pairwise_L1": 1.9840, "ordinary_mean_HHI": 0.5289, "ordinary_mean_cross_seed_HHI_sd": 0.1785,
     "ordinary_mean_asset_cross_seed_sd": 0.0853, "ordinary_max_asset_cross_seed_sd": 0.2203},
])


def load_raw() -> pd.DataFrame:
    raw = pd.read_csv(OUT / f"{PREFIX}_seed_stability_diagnostic.csv")
    # seed 0 was saved as YYYY-MM-DD, later seeds with a time of day
    raw["date"] = pd.to_datetime(raw["date"], format="mixed", errors="raise").dt.normalize()
    assert len(raw) == 1600, f"expected 1600 budget vectors, found {len(raw)}"
    for variant in VARIANTS:
        g = raw[raw["variant"] == variant]
        assert sorted(g["seed"].unique().tolist()) == SEEDS
        assert (g.groupby("seed").size() == 80).all() and (g.groupby("date").size() == 10).all()
    budgets = raw[BUDGET_COLS].to_numpy(dtype=float)
    assert np.allclose(budgets.sum(axis=1), 1.0, atol=1e-6) and budgets.min() >= FLOOR - 1e-6
    return raw


def stability_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        g = raw[raw["variant"] == variant]
        pairwise_l1, agreement = [], []
        for _, gd in g.groupby("date"):
            b = gd.sort_values("seed")[BUDGET_COLS].to_numpy(dtype=float)
            pairwise_l1 += [float(np.abs(b[i] - b[j]).sum()) for i, j in itertools.combinations(range(10), 2)]
            agreement.append(float(np.bincount(np.argmax(b, axis=1), minlength=10).max() / 10))
        asset_sd = g.groupby("date")[BUDGET_COLS].std(ddof=1).mean()
        hhi_sd = g.groupby("date")["HHI"].std(ddof=1)
        rows.append({
            "floor": FLOOR, "epochs": 200, "variant": variant, "n_seeds": g["seed"].nunique(),
            "n_dates": g["date"].nunique(), "n_pairwise_comparisons": len(pairwise_l1),
            "mean_pairwise_L1": float(np.mean(pairwise_l1)), "median_pairwise_L1": float(np.median(pairwise_l1)),
            "max_pairwise_L1": float(np.max(pairwise_l1)), "mean_HHI": float(g["HHI"].mean()),
            "sd_HHI_all_vectors": float(g["HHI"].std(ddof=1)), "mean_cross_seed_HHI_sd": float(hhi_sd.mean()),
            "mean_asset_cross_seed_sd": float(asset_sd.mean()), "max_asset_cross_seed_sd": float(asset_sd.max()),
            "max_sd_asset": asset_sd.idxmax().replace("budget_", ""),
            "mean_dominant_asset_agreement": float(np.mean(agreement)),
            "mean_floor_hits_per_vector": float(g["n_floor_hits"].mean()),
            "fraction_vectors_with_floor_hit": float((g["n_floor_hits"] > 0).mean()),
            "mean_max_budget": float(g["max_budget"].mean()), "max_budget": float(g["max_budget"].max()),
            "mean_min_budget": float(g["min_budget"].mean()),
        })
    return pd.DataFrame(rows)


def comparison(summary: pd.DataFrame) -> pd.DataFrame:
    c = summary.merge(ORDINARY, on="variant", how="left")
    for name, bounded, ordinary in [
        ("L1_reduction_fraction", "mean_pairwise_L1", "ordinary_mean_pairwise_L1"),
        ("asset_sd_reduction_fraction", "mean_asset_cross_seed_sd", "ordinary_mean_asset_cross_seed_sd"),
        ("HHI_sd_reduction_fraction", "mean_cross_seed_HHI_sd", "ordinary_mean_cross_seed_HHI_sd"),
        ("max_asset_sd_reduction_fraction", "max_asset_cross_seed_sd", "ordinary_max_asset_cross_seed_sd"),
    ]:
        c[name] = 1.0 - c[bounded] / c[ordinary]
    return c


def max_difference(computed: pd.DataFrame, saved: pd.DataFrame) -> float:
    numeric = [c for c in saved.columns if pd.api.types.is_numeric_dtype(saved[c])]
    assert list(computed["variant"]) == list(saved["variant"]) and set(numeric) <= set(computed.columns)
    return float((computed[numeric].to_numpy(dtype=float) - saved[numeric].to_numpy(dtype=float)).__abs__().max())


def main() -> int:
    raw = load_raw()
    summary = stability_summary(raw)
    compared = comparison(summary)

    ordinary_file = pd.read_csv(OUT / "lstm_seed_stability_summary.csv").set_index("variant")
    ref = ORDINARY.set_index("variant")
    reference_gap = max(abs(ordinary_file.loc[v, "mean_pairwise_L1_across_seeds"] - ref.loc[v, "ordinary_mean_pairwise_L1"])
                        for v in VARIANTS)
    reference_gap = max(reference_gap, *(abs(ordinary_file.loc[v, "mean_asset_cross_seed_sd"] - ref.loc[v, "ordinary_mean_asset_cross_seed_sd"])
                                         for v in VARIANTS))

    gap_summary = max_difference(summary, pd.read_csv(OUT / f"{PREFIX}_seed_stability_summary.csv"))
    gap_compare = max_difference(compared, pd.read_csv(OUT / f"{PREFIX}_vs_ordinary_softmax.csv"))

    pd.set_option("display.width", 200)
    table4 = pd.DataFrame({
        "Model": ["Baseline LSTM", "Baseline LSTM", "Regime LSTM", "Regime LSTM"],
        "Softmax": ["Ordinary", "Bounded, 5%", "Ordinary", "Bounded, 5%"],
        "Mean pairwise L1": [compared.loc[0, "ordinary_mean_pairwise_L1"], compared.loc[0, "mean_pairwise_L1"],
                             compared.loc[1, "ordinary_mean_pairwise_L1"], compared.loc[1, "mean_pairwise_L1"]],
        "Mean cross-asset SD": [compared.loc[0, "ordinary_mean_asset_cross_seed_sd"], compared.loc[0, "mean_asset_cross_seed_sd"],
                                compared.loc[1, "ordinary_mean_asset_cross_seed_sd"], compared.loc[1, "mean_asset_cross_seed_sd"]],
        "Mean cross-seed HHI SD": [compared.loc[0, "ordinary_mean_cross_seed_HHI_sd"], compared.loc[0, "mean_cross_seed_HHI_sd"],
                                   compared.loc[1, "ordinary_mean_cross_seed_HHI_sd"], compared.loc[1, "mean_cross_seed_HHI_sd"]],
    })
    print("Table 4. Ten-seed LSTM risk-budget stability under ordinary and bounded Softmax\n")
    print(table4.round(3).to_string(index=False))
    print("\nReduction from the 5% floor (baseline, regime):")
    for col in ("L1_reduction_fraction", "asset_sd_reduction_fraction", "HHI_sd_reduction_fraction"):
        print(f"  {col}: " + ", ".join(f"{100 * v:.1f}%" for v in compared[col]))
    print(f"\nMean share of budgets at the 5% floor: baseline {raw[raw.variant == 'baseline'].n_floor_hits.mean() / 10:.1%}, "
          f"regime {raw[raw.variant == 'regime'].n_floor_hits.mean() / 10:.1%}")

    print(f"\nlargest difference from the committed {PREFIX}_seed_stability_summary.csv: {gap_summary:.2e}")
    print(f"largest difference from the committed {PREFIX}_vs_ordinary_softmax.csv:  {gap_compare:.2e}")
    print(f"ordinary-Softmax reference vs outputs/lstm_seed_stability_summary.csv:  {reference_gap:.2e} (rounding)")
    ok = gap_summary < 1e-9 and gap_compare < 1e-9 and reference_gap < 5e-5
    print("PASS: Table 4 reproduced from the saved budgets" if ok else "FAIL: Table 4 not reproduced")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
