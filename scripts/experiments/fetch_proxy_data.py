"""
Trial 16, data step: download the ten long-history proxies and FRED DGS3MO, record their
provenance, and apply the pre-registered futures rule on the overlap with the ETFs
(docs/experiments/trend_blend_proxy_preregistration.md). No strategy is run here.

Writes the raw series to data/proxy/ (not committed, like data/raw/) and, tracked:
  outputs/experiments/proxy_data_manifest.json   download date, rows, first/last date, SHA-256
  outputs/experiments/proxy_overlap_check.csv     each proxy against its ETF, 2007-04-12 to 2026-08-31
  outputs/experiments/proxy_futures_treatment.json  the rule's choice for GC=F and ^SPGSCI

Usage:
    python scripts/experiments/fetch_proxy_data.py
"""

import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data import UNIVERSE, build_risk_free_daily  # noqa: E402
from run_provenance import file_sha256  # noqa: E402

PROXY_DIR = ROOT / "data" / "proxy"
OUT = ROOT / "outputs" / "experiments"
PROXIES = {  # ETF -> (Yahoo ticker, column)
    "SPY": ("VFINX", "Adj Close"), "EFA": ("VGTSX", "Adj Close"), "EEM": ("VEIEX", "Adj Close"),
    "IEF": ("VFITX", "Adj Close"), "TLT": ("VUSTX", "Adj Close"), "LQD": ("VWESX", "Adj Close"),
    "HYG": ("VWEHX", "Adj Close"), "GLD": ("GC=F", "Close"), "DBC": ("^SPGSCI", "Close"), "VNQ": ("VGSIX", "Adj Close"),
}
FUTURES = ["GLD", "DBC"]
OVERLAP = (pd.Timestamp("2007-04-12"), pd.Timestamp("2026-08-31"))
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"


def file_name(etf: str) -> Path:
    return PROXY_DIR / f"{etf}_{PROXIES[etf][0].replace('^', '').replace('=', '')}.csv"


def download() -> None:
    import yfinance as yf

    PROXY_DIR.mkdir(parents=True, exist_ok=True)
    for etf, (ticker, col) in PROXIES.items():
        h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        s = h[col].dropna().rename("value")
        s.index.name = "date"
        s.to_csv(file_name(etf))
        print(f"{etf}: {ticker} {col}, {len(s)} rows, {s.index.min().date()} to {s.index.max().date()}")
    with urllib.request.urlopen(FRED_URL, timeout=60) as r:
        (PROXY_DIR / "DGS3MO.csv").write_bytes(r.read())


def load(etf: str) -> pd.Series:
    return pd.read_csv(file_name(etf), index_col=0, parse_dates=True)["value"]


def main() -> None:
    download()
    files = [file_name(e) for e in PROXIES] + [PROXY_DIR / "DGS3MO.csv"]
    manifest = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance (proxies), FRED fredgraph.csv (DGS3MO)",
        "series": {e: {"ticker": t, "column": c} for e, (t, c) in PROXIES.items()},
        "files": {},
    }
    for f in files:
        df = pd.read_csv(f, index_col=0)
        manifest["files"][f.name] = {"rows": len(df), "first": str(df.index[0]), "last": str(df.index[-1]), "sha256": file_sha256(f)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proxy_data_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Overlap check against the ETFs (data construction only; no strategy)
    yields = pd.read_csv(PROXY_DIR / "DGS3MO.csv", index_col=0, parse_dates=True).iloc[:, 0]
    yields = pd.to_numeric(yields, errors="coerce")
    rows, treatment = [], {}
    for etf in UNIVERSE:
        etf_px = pd.read_csv(ROOT / "data" / "raw" / f"{etf}.csv", index_col=0, parse_dates=True)["AdjClose"]
        both = pd.concat([etf_px.rename("etf"), load(etf).rename("proxy")], axis=1, join="inner").loc[OVERLAP[0] - pd.Timedelta(days=10):OVERLAP[1]]
        ret = both.pct_change().dropna().loc[OVERLAP[0]:OVERLAP[1]]
        rf = build_risk_free_daily(yields, ret.index)["rf_daily"].to_numpy()
        d_as_is = 252 * float((ret.proxy - ret.etf).mean())
        d_plus_rf = 252 * float((ret.proxy + rf - ret.etf).mean())
        row = {"etf": etf, "proxy": PROXIES[etf][0], "overlap_days": len(ret), "daily_correlation": float(ret.corr().iloc[0, 1]),
               "annual_mean_diff_as_is": d_as_is, "annual_mean_diff_plus_rf": d_plus_rf,
               "tracking_error_annual": float((ret.proxy - ret.etf).std(ddof=1) * np.sqrt(252))}
        if etf in FUTURES:
            treatment[etf] = "plus_rf" if abs(d_plus_rf) < abs(d_as_is) else "as_is"
            row["futures_treatment"] = treatment[etf]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "proxy_overlap_check.csv", index=False, float_format="%.6g")
    (OUT / "proxy_futures_treatment.json").write_text(json.dumps(treatment, indent=2))
    pd.set_option("display.width", 200)
    print("\nProxy against ETF, overlap 2007-04-12 to 2026-08-31 (annualised, decimal):")
    print(table.round(4).to_string(index=False))
    print("\nFutures treatment chosen by the pre-registered rule:", treatment)


if __name__ == "__main__":
    main()
