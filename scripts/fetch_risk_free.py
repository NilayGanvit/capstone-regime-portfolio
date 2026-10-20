"""
Build data/raw/risk_free_DGS3MO_daily.csv, the risk-free series every Sharpe
ratio in this project is measured against.

Source: FRED series DGS3MO (market yield on 3-month U.S. Treasury securities,
constant maturity, annualised percent), downloaded from
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO. FRED leaves
holidays blank and can revise past observations, so the download date matters
for exact reproduction; the file written here records none of that, so keep the
download date in the run notes.

Convention (data.build_risk_free_daily): for each ETF return date, the last
yield observed strictly BEFORE that date, compounded to a daily rate over 252
periods, (1 + y/100) ** (1/252) - 1. The file is written on the return
calendar of the ETF files in data/raw/, which is the calendar load_risk_free_daily
requires.

Usage:
    python scripts/fetch_risk_free.py            # download from FRED
    python scripts/fetch_risk_free.py --fred-csv path/to/DGS3MO.csv
"""

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import PriceDataset, RISK_FREE_FILE, build_risk_free_daily  # noqa: E402

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fred-csv", help="use a previously downloaded FRED CSV instead of downloading")
    args = parser.parse_args()

    if args.fred_csv:
        raw = Path(args.fred_csv).read_text()
    else:
        with urllib.request.urlopen(FRED_URL, timeout=60) as response:
            raw = response.read().decode()
    yields = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)["DGS3MO"]

    data_dir = ROOT / "data" / "raw"
    dates = PriceDataset.from_csv_dir(str(data_dir)).returns.index
    frame = build_risk_free_daily(yields, dates)
    out = data_dir / RISK_FREE_FILE
    frame.rename_axis("date").to_csv(out, float_format="%.12f")
    print(f"Wrote {len(frame)} rows ({frame.index.min().date()} to {frame.index.max().date()}) to {out}")
    print(f"FRED observations: {int(yields.notna().sum())}, last {yields.dropna().index.max().date()}")


if __name__ == "__main__":
    main()
