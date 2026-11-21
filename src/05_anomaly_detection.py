"""Step 7: anomaly detection (hourly points AND daily windows).

Methods (all unsupervised, all per household so "normal" is fair for each home):
  * Isolation Forest (primary ML method) - isolates points that are easy to separate
  * Local Outlier Factor (second ML method) - compares each point with its neighbours
  * Z-score per household x hour-of-day   - statistical baseline
  * IQR per household x hour-of-day       - statistical baseline
  * Global z-score (all homes together)   - naive baseline, shown only for comparison

contamination = the share of points we EXPECT to be anomalous. We use 1%
(a flag should be rare) and check 0.5% and 2% as a sensitivity test.
An anomaly flag only means "differs from the learned normal pattern". It is not
proof of theft, faults or any specific cause.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from utils import PROCESSED, APP_DATA, RANDOM_STATE, Z_THRESHOLD, ensure_dirs

CONTAMINATION = 0.01
SENSITIVITY = [0.005, 0.01, 0.02]
LOF_NEIGHBORS = 35
MODEL_FEATURES = ["kwh_s", "robust_z", "weather_z", "diff_s", "hour_sin", "hour_cos", "is_weekend", "temp_z", "flat_run"]
METHODS = ["if", "lof", "z", "iqr"]          # the four methods that vote in the consensus


def build_model_inputs(g):
    """Numeric inputs for one household. No NaN/inf. Clipped so one huge spike cannot dominate."""
    x = pd.DataFrame(index=g.index)
    x["kwh_s"] = (g["kwh"] - g["kwh"].median()) / (g["kwh"].std() + 1e-6)
    x["robust_z"] = g["robust_z"].clip(-30, 30)
    x["weather_z"] = g["weather_z"].clip(-30, 30)
    x["diff_s"] = (g["diff_1"].fillna(0) / g["base_scale"]).clip(-30, 30)
    x["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
    x["is_weekend"] = g["is_weekend"]
    x["flat_run"] = g["flat_run"].clip(0, 12)
    x["temp_z"] = (g["temp_c"] - g["temp_c"].mean()) / (g["temp_c"].std() + 1e-6)
    x = x[MODEL_FEATURES].astype("float64")
    assert np.isfinite(x.to_numpy()).all()
    return StandardScaler().fit_transform(x)


def top_share_flag(score, contamination):
    """Flag the `contamination` share of points with the highest anomaly score."""
    return score >= np.quantile(score, 1 - contamination)


def ml_scores(df):
    """Isolation Forest and LOF anomaly scores per household (higher = more unusual)."""
    if_score = pd.Series(np.nan, index=df.index)
    lof_score = pd.Series(np.nan, index=df.index)
    for hid, g in df.groupby("household_id"):
        X = build_model_inputs(g)
        forest = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(X)
        if_score.loc[g.index] = -forest.score_samples(X)
        lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, n_jobs=-1).fit(X)
        lof_score.loc[g.index] = -lof.negative_outlier_factor_
    return if_score, lof_score


def statistical_flags(df):
    """Z-score and IQR per household x hour-of-day, plus a global z-score."""
    keys = ["household_id", "hour"]
    grp = df.groupby(keys)["kwh"]
    z = (df["kwh"] - grp.transform("mean")) / (grp.transform("std") + 1e-6)
    q1, q3 = grp.transform(lambda s: s.quantile(0.25)), grp.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    z_global = (df["kwh"] - df["kwh"].mean()) / df["kwh"].std()
    flags = (z.abs() > 3), ((df["kwh"] < q1 - 1.5 * iqr) | (df["kwh"] > q3 + 1.5 * iqr)), (z_global.abs() > 3)
    # continuous scores (higher = more unusual); used for the threshold-free AUC in step 7 of the evaluation
    outside = np.maximum(q1 - df["kwh"], df["kwh"] - q3) / (iqr + 1e-6)
    return flags + (z.abs(), outside, z_global.abs())


def reason_tags(df):
    """Plain-language reason for each flagged row (rule based, easy to explain)."""
    near_zero = (df["kwh"] < 0.02) & (df["base_med"] > 0.15)
    high = df["robust_z"] > Z_THRESHOLD
    conditions = [
        near_zero,
        high & (df["weather_z"] <= Z_THRESHOLD),
        high,
        df["robust_z"] < -Z_THRESHOLD,
        (df["diff_1"].abs() / df["base_scale"]) > Z_THRESHOLD,
    ]
    labels = [
        "Near zero while this home usually uses power at this hour",
        "High for this hour, but typical for this temperature",
        "Much higher than usual for this hour, even allowing for temperature",
        "Much lower than usual for this hour",
        "Sudden jump or drop compared with the previous hour",
    ]
    return pd.Series(np.select(conditions, labels, default="Unusual mix of load, time and weather"), index=df.index)


def detect(df, contamination=CONTAMINATION):
    """Run every method on a baselined hourly table. Returns the table with flags added."""
    df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)
    if_score, lof_score = ml_scores(df)
    df["if_score"] = if_score.astype("float32")
    df["lof_score"] = lof_score.astype("float32")
    df["flag_if"] = df.groupby("household_id")["if_score"].transform(lambda s: top_share_flag(s, contamination))
    df["flag_lof"] = df.groupby("household_id")["lof_score"].transform(lambda s: top_share_flag(s, contamination))
    (df["flag_z"], df["flag_iqr"], df["flag_global"],
     df["z_score"], df["iqr_score"], df["global_score"]) = statistical_flags(df)
    df["consensus"] = df[["flag_if", "flag_lof", "flag_z", "flag_iqr"]].sum(axis=1).astype("int8")
    any_flag = df["consensus"] > 0
    df["reason"] = "Normal"
    df.loc[any_flag, "reason"] = reason_tags(df[any_flag])
    return df


def sensitivity_table(df):
    """How many Isolation Forest flags at different contamination values (no refit needed)."""
    rows = []
    for c in SENSITIVITY:
        flags = df.groupby("household_id")["if_score"].transform(lambda s: top_share_flag(s, c))
        rows.append(dict(contamination=c, flagged_hours=int(flags.sum()), share_pct=round(100 * flags.mean(), 2)))
    return pd.DataFrame(rows)


def daily_anomalies(df):
    """Daily-window anomalies: days whose total or shape is unusual for that household."""
    d = df.copy()
    d["date"] = d["timestamp"].dt.floor("D")
    day = d.groupby(["household_id", "city", "date"], observed=True).agg(
        n_hours=("kwh", "size"), total_kwh=("kwh", "sum"), peak_kwh=("kwh", "max"),
        min_kwh=("kwh", "min"), std_kwh=("kwh", "std"), mean_temp=("temp_c", "mean")).reset_index()
    day = day[day["n_hours"] >= 20].copy()            # only almost-complete days
    day["total_kwh"] = day["total_kwh"] / day["n_hours"] * 24     # scale to a full 24 h
    day["std_kwh"] = day["std_kwh"].fillna(0)
    feats = ["total_kwh", "peak_kwh", "min_kwh", "std_kwh", "mean_temp"]
    day["daily_score"] = np.nan
    for hid, g in day.groupby("household_id"):
        X = StandardScaler().fit_transform(g[feats].astype("float64"))
        forest = IsolationForest(n_estimators=200, random_state=RANDOM_STATE).fit(X)
        day.loc[g.index, "daily_score"] = -forest.score_samples(X)
    day["flag_daily"] = day.groupby("household_id")["daily_score"].transform(lambda s: top_share_flag(s, 0.02))
    med = day.groupby("household_id")["total_kwh"].transform("median")
    mad = day.groupby("household_id")["total_kwh"].transform(lambda s: np.median(np.abs(s - np.median(s))))
    day["daily_total_z"] = (day["total_kwh"] - med) / np.maximum(1.4826 * mad, 0.05 * med + 0.01)
    day["daily_reason"] = np.select(
        [day["daily_total_z"] > Z_THRESHOLD, day["daily_total_z"] < -Z_THRESHOLD],
        ["Unusually high total use for this home", "Unusually low total use for this home"],
        default="Unusual daily shape (peak, minimum or variation)")
    day.loc[~day["flag_daily"], "daily_reason"] = "Normal"
    return day


def main():
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "baselined.parquet")
    print(f"Detecting on {len(df):,} hourly rows from {df.household_id.nunique()} households ...")
    out = detect(df)
    print("\nFlag rate per method (%):")
    for c in ["flag_if", "flag_lof", "flag_z", "flag_iqr", "flag_global"]:
        print(f"  {c:<12} {out[c].mean():.2%}")
    print("\nConsensus (how many of the 4 methods agree):")
    print(out["consensus"].value_counts().sort_index().to_string())
    print("\nReason tags of flagged hours:")
    print(out.loc[out.consensus > 0, "reason"].value_counts().to_string())

    sens = sensitivity_table(out)
    print("\nSensitivity (Isolation Forest):\n" + sens.to_string(index=False))
    day = daily_anomalies(out)
    print(f"\nDaily windows: {len(day):,}  flagged: {int(day.flag_daily.sum())}")

    out.to_parquet(PROCESSED / "anomalies_full.parquet", index=False)
    keep = ["household_id", "city", "timestamp", "kwh", "temp_c", "humidity", "precip_mm", "hour", "month",
            "is_weekend", "base_med", "robust_z", "weather_med", "weather_z", "if_score", "lof_score",
            "flag_if", "flag_lof", "flag_z", "flag_iqr", "flag_global", "consensus", "reason"]
    app = out[keep].copy()
    app["reason"] = app["reason"].astype("category")
    for c in ["base_med", "robust_z", "weather_med", "weather_z", "if_score", "lof_score"]:
        app[c] = app[c].astype("float32")
    app.to_parquet(APP_DATA / "anomalies.parquet", index=False)
    day.to_parquet(APP_DATA / "daily_anomalies.parquet", index=False)
    sens.to_csv(APP_DATA / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
