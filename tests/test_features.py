import pandas as pd

def test_feature_and_target_shapes():
    df = pd.read_csv("data/processed/v0/train.csv", parse_dates=["timestamp"])
    assert {"rolling_avg_10","volume_sum_10","target","stock_id","timestamp"}.issubset(df.columns)
    assert df["rolling_avg_10"].notna().sum() > 10
    assert df["volume_sum_10"].notna().sum() > 10
    assert set(df["target"].dropna().unique()).issubset({0,1})
