#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

FEATURES = ["rolling_avg_10", "volume_sum_10"]
TARGET = "target"
ID_COLS = ["timestamp", "stock_id"]


def log(msg: str):
    print(f"[TRAIN] {msg}", flush=True)


def load_processed(version_dir: Path):
    log(f"Loading processed data from: {version_dir}")
    train_pq = version_dir / "train.parquet"
    test_pq = version_dir / "test.parquet"

    if train_pq.exists() and test_pq.exists():
        train = pd.read_parquet(train_pq)
        test = pd.read_parquet(test_pq)
        log("Loaded Parquet files.")
    else:
        train_csv = version_dir / "train.csv"
        test_csv = version_dir / "test.csv"
        if not train_csv.exists():
            raise FileNotFoundError(f"Missing {train_csv}")
        if not test_csv.exists():
            raise FileNotFoundError(f"Missing {test_csv}")
        train = pd.read_csv(train_csv, parse_dates=["timestamp"])
        test = pd.read_csv(test_csv, parse_dates=["timestamp"])
        log("Loaded CSV files.")

    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    log(f"Train rows: {len(train)}, Test rows: {len(test)}")
    return train, test


def feast_align_features(repo_path: str, df: pd.DataFrame) -> pd.DataFrame:
    log("Aligning features via Feast…")
    store = FeatureStore(repo_path=repo_path)
    entity_df = df[ID_COLS].rename(columns={"timestamp": "event_timestamp"}).copy()
    feast_features = [f"stock_features:{f}" for f in FEATURES]
    hist = store.get_historical_features(entity_df=entity_df, features=feast_features).to_df()

    # Normalize columns
    hist = hist.rename(columns={"event_timestamp": "timestamp"})
    rename_map = {}
    for c in hist.columns:
        if "__" in c:
            base = c.split("__", 1)[-1]
            if base in FEATURES:
                rename_map[c] = base
    if rename_map:
        hist = hist.rename(columns=rename_map)

    keep_cols = [c for c in (ID_COLS + FEATURES) if c in hist.columns] + ID_COLS
    hist = hist[keep_cols].drop_duplicates(ID_COLS, keep="last")

    out = df.merge(hist, on=ID_COLS, how="left", suffixes=("", "_feast"))
    for f in FEATURES:
        feast_col = f  # post-rename
        if f + "_feast" in out.columns:
            out[f] = out[f + "_feast"].fillna(out[f])
            out.drop(columns=[f + "_feast"], inplace=True)
        elif feast_col in out.columns and f not in out.columns:
            out[f] = out[feast_col]
    log("Feast alignment complete.")
    return out


def train_and_log(train_df: pd.DataFrame, test_df: pd.DataFrame, run_name: str, use_feast: bool):
    log(f"Training run: {run_name} (use_feast={use_feast})")
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values.astype(int)
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values.astype(int)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 8],
        "min_samples_split": [2, 5],
        "random_state": [42],
        "n_jobs": [-1],
    }

    best = {"score": -1, "params": None, "model": None}
    for params in ParameterGrid(param_grid):
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, proba) if proba is not None and len(np.unique(y_test)) > 1 else float("nan")

        with mlflow.start_run(run_name=run_name):
            log_params = dict(params)
            log_params["use_feast"] = use_feast
            log_params["train_rows"] = len(train_df)
            log_params["test_rows"] = len(test_df)
            log_params["features"] = ",".join(FEATURES)
            mlflow.log_params(log_params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            if not np.isnan(auc):
                mlflow.log_metric("auc", auc)

            Path("models").mkdir(exist_ok=True)
            model_path = Path("models") / f"rf_{int(datetime.utcnow().timestamp())}.joblib"
            joblib.dump(clf, model_path)
            mlflow.log_artifact(str(model_path))
            log(f"Logged model: {model_path.name}  acc={acc:.4f}  f1={f1:.4f}  auc={(auc if not np.isnan(auc) else 'nan')}")

        if acc > best["score"]:
            best.update({"score": acc, "params": params, "model": clf})

    log(f"Best accuracy: {best['score']:.4f} with {best['params']}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0", help="Processed version folder name, e.g., v0 or v0+v1")
    parser.add_argument("--use-feast", action="store_true", help="Fetch/align features via Feast")
    parser.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "file:mlflow_backend"))
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    exp_name = "stock_exam"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    log(f"MLflow tracking: {args.mlflow_uri}  experiment: {exp_name}")

    version_dir = Path("data/processed") / args.version
    train_df, test_df = load_processed(version_dir)

    if args.use_feast:
        train_df = feast_align_features("feature_repo", train_df)
        test_df = feast_align_features("feature_repo", test_df)

    keep = ID_COLS + FEATURES + [TARGET]
    train_df = train_df[keep].dropna().copy()
    test_df = test_df[keep].dropna().copy()
    log(f"Using columns: {keep}")
    log(f"Train rows after dropna: {len(train_df)}; Test rows after dropna: {len(test_df)}")

    best = train_and_log(train_df, test_df, run_name=f"rf_{args.version}", use_feast=args.use_feast)

    preds = best["model"].predict(test_df[FEATURES].values)
    out = test_df[ID_COLS + [TARGET]].copy()
    out["pred"] = preds
    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / f"preds_{args.version}.csv"
    out.to_csv(out_path, index=False)
    log(f"Saved predictions to {out_path}")

    print("[OK] Training complete")
    print(" Best params:", best["params"])
    print(" Best accuracy:", round(best["score"], 4))
    print(" Predictions:", str(out_path))


if __name__ == "__main__":
    main()
