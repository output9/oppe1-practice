#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
import pandas as pd


def infer_stock_id_from_filename(path: Path) -> str:
    """
    Extract a stock_id from filenames like:
      AARTIIND__EQ__NSE__NSE__MINUTE.csv  -> AARTIIND
      ABCAPITAL_EQNSENSE_MINUTE.csv       -> ABCAPITAL (fallback)
    """
    stem = path.stem  # filename without .csv
    # Prefer the first token until first underscore/double-underscore
    m = re.match(r"([A-Z0-9]+)", stem)
    if m:
        return m.group(1)
    return stem.split("_")[0]


def load_and_prepare_one_file(
    fpath: Path,
    horizon_min: int = 5,
    lookback_min: int = 10,
    sample_n: int = None,
) -> pd.DataFrame:
    """
    Load one raw CSV:
      columns: timestamp,open,high,low,close,volume
    - Parse timestamp
    - Sort (no ordering assumption)
    - Fill per-minute gaps with forward fill (as required)
    - Compute features:
        rolling_avg_10 (close rolling 10-min mean)
        volume_sum_10  (volume rolling 10-min sum)
    - Create binary target: close(t+5) > close(t)
    - Drop cold-start (first lookback-1 mins) and hot-end (last horizon mins)
    - Optionally sample first N rows (post feature creation) for fast mode
    - Add stock_id column from filename
    """
    df = pd.read_csv(fpath)
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(map(str.lower, df.columns))
    # Make columns lowercase to be robust
    df.columns = [c.lower() for c in df.columns]
    if missing:
        raise ValueError(f"{fpath.name}: missing required columns {missing}")

    # Parse timestamp -> datetime and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Set index and ensure per-minute continuity (resample 1T)
    df = df.set_index("timestamp")
    # Resample to 1-minute grid; use forward-fill to "augment with previous minute’s data values"
    df = df.resample("1T").ffill()

    # Features
    df["rolling_avg_10"] = df["close"].rolling(window=lookback_min, min_periods=lookback_min).mean()
    df["volume_sum_10"]  = df["volume"].rolling(window=lookback_min, min_periods=lookback_min).sum()

    # Target: future close after +horizon minutes
    df["future_close"] = df["close"].shift(-horizon_min)
    df["target"] = (df["future_close"] > df["close"]).astype("Int64")

    # Drop rows that don’t have enough history or future
    df = df.dropna(subset=["rolling_avg_10", "volume_sum_10", "future_close"])

    # Optional: limit to first N rows post-feature creation
    if sample_n is not None and sample_n > 0:
        df = df.iloc[:sample_n].copy()

    # Add stock_id
    stock_id = infer_stock_id_from_filename(fpath)
    df["stock_id"] = stock_id

    # Keep only needed columns
    df = df.reset_index()  # bring timestamp back as column
    cols = ["timestamp", "stock_id", "open", "high", "low", "close", "volume",
            "rolling_avg_10", "volume_sum_10", "target"]
    df = df[cols]

    return df


def load_all_from_version(
    version_dir: Path,
    horizon_min: int,
    lookback_min: int,
    sample_n: int,
) -> pd.DataFrame:
    csvs = sorted(version_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {version_dir}")

    dfs = []
    for f in csvs:
        try:
            dfi = load_and_prepare_one_file(
                f, horizon_min=horizon_min, lookback_min=lookback_min, sample_n=sample_n
            )
            dfs.append(dfi)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")

    if not dfs:
        raise RuntimeError(f"No usable files in {version_dir}")

    return pd.concat(dfs, ignore_index=True)


def time_based_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global chronological split across all stocks (no shuffle).
    Ensures we simulate 'train on past, test on future'.
    """
    df = df.sort_values(["timestamp", "stock_id"]).reset_index(drop=True)
    n = len(df)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    train = df.iloc[:n_train].copy()
    test  = df.iloc[n_train:].copy()
    return train, test


def main():
    ap = argparse.ArgumentParser(description="Preprocess raw minute data and create features/labels.")
    ap.add_argument("--input-root", default="data_versions", help="Root folder containing v0, v1, etc.")
    ap.add_argument("--version", nargs="+", required=True, help="One or more versions to include, e.g., v0 or v0 v1")
    ap.add_argument("--output-root", default="data/processed", help="Where to write processed outputs.")
    ap.add_argument("--horizon", type=int, default=5, help="Prediction horizon in minutes (default 5).")
    ap.add_argument("--lookback", type=int, default=10, help="Lookback window in minutes for features (default 10).")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split (default 0.2).")
    ap.add_argument("--sample-n", type=int, default=None, help="If set, limit to first N rows per file for fast mode.")
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    # Aggregate across selected versions
    all_df = []
    for v in args.version:
        vdir = input_root / v
        print(f"[INFO] Loading version: {vdir}")
        dfv = load_all_from_version(vdir, args.horizon, args.lookback, args.sample_n)
        all_df.append(dfv)

    df = pd.concat(all_df, ignore_index=True)

    # Split
    train_df, test_df = time_based_split(df, test_size=args.test_size)

    # Output dirs
    out_dir = output_root / "+".join(args.version)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save
    train_path = out_dir / "train.csv"
    test_path  = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Summary
    summary = {
        "versions": args.version,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "stocks": sorted(df["stock_id"].unique().tolist()),
        "horizon_min": args.horizon,
        "lookback_min": args.lookback,
        "test_size": args.test_size,
        "sample_n_per_file": args.sample_n,
        "columns": train_df.columns.tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[OK] Wrote: {train_path} ({len(train_df)} rows), {test_path} ({len(test_df)} rows)")
    print(f"[OK] Summary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
