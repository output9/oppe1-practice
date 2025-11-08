#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

FEATURES = ["rolling_avg_10", "volume_sum_10"]
TARGET = "target"

def load_processed(version_dir: Path):
    test_pq  = version_dir / "test.parquet"
    if not test_pq.exists():
        return pd.read_csv(version_dir / "test.csv", parse_dates=["timestamp"])
    return pd.read_parquet(test_pq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0")
    ap.add_argument("--model-path", required=False, help="Path to a .joblib model; if not set, uses latest in models/")
    args = ap.parse_args()

    version_dir = Path("data/processed") / args.version
    test_df = load_processed(version_dir)
    test_df = test_df.dropna(subset=FEATURES+[TARGET]).copy()

    # Pick latest model if not provided
    if not args.model_path:
        models = sorted(Path("models").glob("rf_*.joblib"))
        if not models:
            raise FileNotFoundError("No models found in models/. Train first.")
        model_path = models[-1]
    else:
        model_path = Path(args.model_path)

    clf = joblib.load(model_path)
    X = test_df[FEATURES].values
    y = test_df[TARGET].values.astype(int)

    preds = clf.predict(X)
    proba = clf.predict_proba(X)[:,1] if hasattr(clf,"predict_proba") else None

    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, proba) if proba is not None and len(np.unique(y))>1 else float("nan")

    print(f"Model: {model_path.name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    if not np.isnan(auc):
        print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
