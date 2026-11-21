"""Step 5: feature engineering.

Why: models need numbers that describe "normal behaviour": when it is (hour, day,
month), what happened just before (lags, rolling stats) and the weather.

No data leakage: rolling and lag features only use PAST hours (shift before rolling).
We rebuild a complete hourly grid per household first, so "lag 1" really means the
previous hour; where an hour is missing the lag is left empty (NaN).
"""
import numpy as np
import pandas as pd

from utils import PROCESSED, PEAK_HOURS


def add_time_features(df):
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["day_of_week"] = ts.dt.day_of_week.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")   # Sat/Sun
    df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype("int8")
    return df


def add_history_features(df):
    """Lag / rolling / change features per household (past values only)."""
    out = []
    for hid, g in df.groupby("household_id", sort=True):
        full = g.set_index("timestamp").sort_index()
        grid = full[["kwh"]].resample("h").mean()                 # complete hourly grid
        past = grid["kwh"].shift(1)
        feats = pd.DataFrame(index=grid.index)
        feats["lag_1"] = grid["kwh"].shift(1)
        feats["lag_24"] = grid["kwh"].shift(24)
        feats["diff_1"] = grid["kwh"] - feats["lag_1"]
        feats["roll24_mean"] = past.rolling(24, min_periods=12).mean()
        feats["roll24_std"] = past.rolling(24, min_periods=12).std()
        g = full.join(feats, how="left").reset_index()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    df = pd.read_parquet(PROCESSED / "hourly_merged.parquet")
    df = add_time_features(df)
    df = add_history_features(df)
    for c in ["lag_1", "lag_24", "diff_1", "roll24_mean", "roll24_std"]:
        df[c] = df[c].astype("float32")
    df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)
    print(f"Rows: {len(df):,}  columns: {len(df.columns)}")
    print("Missing share per feature (first hours / gaps are expected):")
    print(df[["lag_1", "lag_24", "diff_1", "roll24_mean", "roll24_std"]].isna().mean().round(3).to_string())
    assert not np.isinf(df.select_dtypes("number")).any().any()
    df.to_parquet(PROCESSED / "features.parquet", index=False)


if __name__ == "__main__":
    main()
