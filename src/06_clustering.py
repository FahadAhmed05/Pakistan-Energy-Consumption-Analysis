"""Step 8: group households with similar consumption behaviour (K-Means) and show them with PCA.

Household features (one row per household, 59 rows):
  mean_kwh        average level (log scale, because homes differ a lot in size)
  peak_to_avg     highest average hour / average of all hours  -> "spikiness"
  night / morning / afternoon / evening share of daily energy   -> daily shape
  weekend_diff    weekend average vs weekday average (%)
  temp_corr       correlation of hourly use with outdoor temperature

K is chosen with the elbow (inertia) and the silhouette score for K = 2..8.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import PROCESSED, APP_DATA, RANDOM_STATE, ensure_dirs

FEATURES = ["log_mean_kwh", "peak_to_avg", "night_share", "morning_share", "afternoon_share",
            "evening_share", "weekend_diff_pct", "temp_corr"]
LABELS = {"log_mean_kwh": "overall use", "peak_to_avg": "peaky use", "night_share": "night use",
          "morning_share": "morning use", "afternoon_share": "afternoon use",
          "evening_share": "evening use", "weekend_diff_pct": "weekend use", "temp_corr": "temperature sensitivity"}


def household_features(df):
    rows = []
    for hid, g in df.groupby("household_id"):
        profile = g.groupby("hour")["kwh"].mean().reindex(range(24)).interpolate()
        total = profile.sum()
        wk = g.loc[g.is_weekend == 1, "kwh"].mean()
        wd = g.loc[g.is_weekend == 0, "kwh"].mean()
        rows.append(dict(
            household_id=hid, city=str(g["city"].iloc[0]), mean_kwh=g["kwh"].mean(),
            log_mean_kwh=np.log(g["kwh"].mean() + 1e-3), peak_to_avg=profile.max() / profile.mean(),
            night_share=profile[0:6].sum() / total, morning_share=profile[6:12].sum() / total,
            afternoon_share=profile[12:18].sum() / total, evening_share=profile[18:24].sum() / total,
            weekend_diff_pct=100 * (wk - wd) / wd, temp_corr=g["kwh"].corr(g["temp_c"]),
            peak_hour=int(profile.idxmax())))
    return pd.DataFrame(rows)


def choose_k(X):
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_)))
    return pd.DataFrame(rows)


def name_cluster(center, overall):
    """Plain-words name from what stands out most in the cluster centre (z-scores)."""
    z = (center - overall["mean"]) / overall["std"]
    top = z[FEATURES].abs().sort_values(ascending=False).index[:2]
    parts = [("high " if z[f] > 0 else "low ") + LABELS[f] for f in top]
    return " & ".join(parts)


def main():
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "baselined.parquet", columns=["household_id", "city", "hour", "kwh", "temp_c", "is_weekend"])
    feat = household_features(df)
    X = StandardScaler().fit_transform(feat[FEATURES])
    ks = choose_k(X)
    print(ks.round(3).to_string(index=False))
    # silhouette favours K=2 on small data; we want useful groups, so pick the best score from K = 3..6
    best_k = int(ks[(ks.k >= 3) & (ks.k <= 6)].sort_values("silhouette", ascending=False).iloc[0]["k"])
    print(f"Chosen K = {best_k}")
    km = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE).fit(X)
    feat["cluster"] = km.labels_

    overall = {"mean": feat[FEATURES].mean(), "std": feat[FEATURES].std()}
    names = {}
    for c in range(best_k):
        members = feat[feat.cluster == c]
        names[c] = f"Group {c + 1}: " + name_cluster(members[FEATURES].mean(), overall)
    feat["cluster_name"] = feat["cluster"].map(names)

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    xy = pca.transform(X)
    feat["pc1"], feat["pc2"] = xy[:, 0], xy[:, 1]
    print(f"PCA variance explained: {pca.explained_variance_ratio_.round(3)}")

    # anomaly rate per cluster (from step 7). The Isolation Forest rate is 1% in every home by design
    # (contamination is set per household), so the fair comparison is how often 2+ methods agree.
    an = pd.read_parquet(APP_DATA / "anomalies.parquet", columns=["household_id", "flag_if", "consensus"])
    rate = an.groupby("household_id").agg(anomaly_rate_if=("flag_if", "mean"), multi_method=("consensus", lambda s: (s >= 2).mean()))
    feat = feat.merge(rate.reset_index(), on="household_id")
    summary = feat.groupby("cluster_name").agg(households=("household_id", "size"), mean_kwh=("mean_kwh", "mean"),
              peak_to_avg=("peak_to_avg", "mean"), evening_share=("evening_share", "mean"),
              night_share=("night_share", "mean"), temp_corr=("temp_corr", "mean"),
              multi_method_rate_pct=("multi_method", lambda s: 100 * s.mean())).round(3).reset_index()
    print(summary.to_string(index=False))
    print(pd.crosstab(feat["cluster_name"], feat["city"]).to_string())

    feat.to_csv(APP_DATA / "household_clusters.csv", index=False)
    ks.to_csv(APP_DATA / "cluster_k_selection.csv", index=False)
    summary.to_csv(APP_DATA / "cluster_summary.csv", index=False)
    pd.DataFrame({"pc": ["PC1", "PC2"], "explained": pca.explained_variance_ratio_}).to_csv(APP_DATA / "pca_variance.csv", index=False)


if __name__ == "__main__":
    main()
