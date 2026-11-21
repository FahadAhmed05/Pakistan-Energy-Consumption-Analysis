"""Step 6: learn each household's NORMAL pattern and measure deviation from it.

Two baselines per household (both use robust statistics: median and MAD, so the
anomalies themselves do not distort what "normal" means):

 1. Time baseline    : normal kWh for (hour of day x weekday/weekend)
 2. Weather baseline : normal kWh for (hour of day x temperature band)
                       -> answers "is this high only because it is hot/cold?"

robust_z  = (kwh - median) / (1.4826 * MAD)     (like a z-score, but robust)
A value around 0 is normal; beyond +-3.5 is unusual (the usual cut-off for this score).

The baseline is fitted on the whole recorded history of the household
(offline analysis). This is the usual setup for finding past anomalies.
"""
import numpy as np
import pandas as pd

from utils import PROCESSED, Z_THRESHOLD

MAD_SCALE = 1.4826     # makes MAD comparable to a standard deviation for normal data
N_TEMP_BANDS = 5       # temperature bands per household (quintiles)
MIN_GROUP = 10         # smallest group size allowed for the weather baseline


def mad(x):
    return np.median(np.abs(x - np.median(x)))


def group_median_mad(df, keys, value="kwh"):
    """Median and MAD of `value` for each combination of `keys`."""
    g = df.groupby(keys, observed=True)[value]
    out = g.median().rename("med").to_frame()
    out["mad"] = df.assign(_m=g.transform("median")).assign(_d=lambda d: (d[value] - d["_m"]).abs()) \
                   .groupby(keys, observed=True)["_d"].median()
    out["n"] = g.size()
    return out.reset_index()


def add_baselines(df):
    """Add base_med, base_mad, robust_z, temp_band, weather_med, weather_mad, weather_z."""
    df = df.copy()
    df["day_type"] = df["is_weekend"]
    # per-household floor for the MAD so a very flat household does not give huge z-scores
    g = df.groupby("household_id")["kwh"]
    house_scale = 0.1 * g.transform("median") + 0.05 * g.transform("std") + 0.01

    # 1) time baseline
    keys = ["household_id", "hour", "day_type"]
    base = group_median_mad(df, keys).rename(columns={"med": "base_med", "mad": "base_mad"}).drop(columns="n")
    df = df.merge(base, on=keys, how="left")
    df["base_scale"] = np.maximum(MAD_SCALE * df["base_mad"], house_scale)
    df["robust_z"] = (df["kwh"] - df["base_med"]) / df["base_scale"]

    # 2) weather baseline (temperature bands are per household, so every band has plenty of data)
    df["temp_band"] = df.groupby("household_id")["temp_c"].transform(
        lambda s: pd.qcut(s, N_TEMP_BANDS, labels=False, duplicates="drop")).astype("int8")
    wkeys = ["household_id", "hour", "temp_band"]
    wb = group_median_mad(df, wkeys).rename(columns={"med": "weather_med", "mad": "weather_mad", "n": "weather_n"})
    df = df.merge(wb, on=wkeys, how="left")
    small = df["weather_n"] < MIN_GROUP            # tiny groups -> fall back to the time baseline
    df.loc[small, "weather_med"] = df.loc[small, "base_med"]
    df.loc[small, "weather_mad"] = df.loc[small, "base_mad"]
    wscale = np.maximum(MAD_SCALE * df["weather_mad"], house_scale)
    df["weather_z"] = (df["kwh"] - df["weather_med"]) / wscale
    return df.drop(columns=["weather_n"])


def main():
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df = add_baselines(df)
    cols = ["base_med", "base_mad", "base_scale", "robust_z", "weather_med", "weather_mad", "weather_z"]
    for c in cols:
        df[c] = df[c].astype("float32")
    assert not df[["robust_z", "weather_z"]].isna().any().any()
    assert not np.isinf(df[cols]).any().any()
    print(f"Rows: {len(df):,}")
    print(df[["robust_z", "weather_z"]].describe().round(2).to_string())
    print(f"|robust_z|>3.5: {(df.robust_z.abs() > Z_THRESHOLD).mean():.2%}   |weather_z|>3.5: {(df.weather_z.abs() > Z_THRESHOLD).mean():.2%}")
    df.to_parquet(PROCESSED / "baselined.parquet", index=False)


if __name__ == "__main__":
    main()
