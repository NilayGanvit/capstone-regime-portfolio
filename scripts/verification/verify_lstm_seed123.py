"""Verify saved Colab LSTM seed-123 and ten-seed results without training."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/content/drive/MyDrive/Capstone_Shared/Code")
FROZEN = ROOT / "frozen_reference_5pct"
VERIFIED = ROOT / "verification_seed123_colab"
TEN_SEED = FROZEN / "ten_seed_robustness"

MODELS = ["lstm_baseline", "lstm_regime", "lstm_blend"]

reference = pd.read_csv(
    FROZEN / "results_seed123/real_data_rq_net_returns_5bps.csv"
)
reproduced = pd.read_csv(
    VERIFIED / "nilay_seed123_colab_comparison.csv"
)

merged = reference[["date"] + MODELS].merge(
    reproduced[["date"] + MODELS],
    on="date",
    how="outer",
    suffixes=("_reference", "_reproduced"),
    indicator=True,
    validate="one_to_one",
)

assert len(merged) == 1926
assert (merged["_merge"] == "both").all()

for model in MODELS:
    a = merged[f"{model}_reference"].to_numpy()
    b = merged[f"{model}_reproduced"].to_numpy()

    assert np.array_equal(a, b), f"Mismatch: {model}"

print("PASS: Seed 123 matches across all 1926 dates and 3 models.")

rf = pd.read_csv(
    FROZEN / "data/raw/risk_free_DGS3MO_daily.csv",
    parse_dates=["date"],
)[["date", "rf_daily"]]

max_difference = 0.0
count = 0

for seed in range(10):
    daily = pd.read_csv(
        TEN_SEED / f"seed_{seed:02d}_net_returns.csv",
        parse_dates=["date"],
    )
    metrics = pd.read_csv(
        TEN_SEED / f"seed_{seed:02d}_metrics.csv"
    )

    aligned = daily.merge(rf, on="date", how="left", validate="one_to_one")

    assert len(aligned) == 1926
    assert aligned["rf_daily"].notna().all()

    for model in MODELS:
        excess = aligned[model] - aligned["rf_daily"]
        calculated = np.sqrt(252) * excess.mean() / excess.std(ddof=1)

        reported = metrics.loc[
            metrics["config"] == model, "sharpe_net_excess"
        ].iloc[0]

        difference = abs(calculated - reported)
        max_difference = max(max_difference, difference)

        assert difference < 1e-10, (
            f"Seed {seed}, {model}: difference={difference}"
        )
        count += 1

print(f"PASS: All {count} ten-seed Sharpe ratios reproduced.")
print(f"Maximum absolute Sharpe difference: {max_difference:.3e}")
