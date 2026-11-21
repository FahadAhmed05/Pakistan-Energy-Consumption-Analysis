"""Step 9: evaluation. The data has NO ground-truth anomaly labels, so we use five checks:

 1. Synthetic injection test  - inject KNOWN anomalies into real data, see which method finds them.
                                (a controlled validation, NOT real labels)
 2. Anomaly-rate analysis     - per method / household / city / month
 3. Agreement between methods - overlap (Jaccard) of what each method flags
 4. Manual inspection         - representative flagged cases and false-alarm-looking cases
 5. Weather check             - do flags pile up in hot weather, and does the weather-aware score help?

Injection design (per household, 3 repeats with different seeds, 12 households = 2 per city):
  spike     12 single hours       kWh + 3..6 x the household's std
  drop      12 single hours       set to 0 (only where the home normally uses >= 0.5 kWh: an outage)
  shift      4 blocks of 6 hours  kWh x 2.5       (sustained level change)
  flatline   4 blocks of 6 hours  stuck at the first value (frozen meter)
Hours that 2+ methods already flagged in the real data are "ambiguous" (they may be genuine
anomalies) so we never inject there and we leave them out of the scoring.
"""
import importlib

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from utils import PROCESSED, APP_DATA, RANDOM_STATE, Z_THRESHOLD, ensure_dirs

feat_mod = importlib.import_module("03_features")
base_mod = importlib.import_module("04_baseline_profiles")
det_mod = importlib.import_module("05_anomaly_detection")

N_SPIKE, N_DROP, N_SHIFT_BLOCKS, N_FLAT_BLOCKS, BLOCK = 12, 12, 4, 4, 6
N_REPEATS = 3
DROP_MIN_NORMAL = 0.5   # a drop to zero only counts where the home normally uses at least this much (kWh)
METHOD_COLS = {"Isolation Forest": "flag_if", "Local Outlier Factor": "flag_lof", "Z-score (per home & hour)": "flag_z",
               "IQR (per home & hour)": "flag_iqr", "Global z-score (naive)": "flag_global", "Consensus (2+ methods)": "flag_cons2"}
SCORE_COLS = {"Isolation Forest": "if_score", "Local Outlier Factor": "lof_score", "Z-score (per home & hour)": "z_score",
              "IQR (per home & hour)": "iqr_score", "Global z-score (naive)": "global_score", "Consensus (2+ methods)": "consensus"}
HISTORY_COLS = ["lag_1", "lag_24", "diff_1", "roll24_mean", "roll24_std", "flat_run"]   # recomputed after injection
TYPES = ["spike", "drop", "shift", "flatline"]


def pick_positions(ok, ts, n, block, taken, rng):
    """Pick n non-overlapping start positions whose `block` consecutive hours are all usable."""
    n_rows = len(ok)
    starts, tries = [], 0
    while len(starts) < n and tries < 20000:
        tries += 1
        i = int(rng.integers(30, n_rows - block - 30))
        span = range(i - 3, i + block + 3)
        if any(j in taken for j in span) or not ok[i:i + block].all():
            continue
        if block > 1 and (ts[i + block - 1] - ts[i]) != np.timedelta64(block - 1, "h"):
            continue
        starts.append(i)
        taken.update(range(i, i + block))
    return starts


def inject(g, ambiguous, rng):
    """Return a copy of one household's rows with injected anomalies and a label column."""
    g = g.sort_values("timestamp").reset_index(drop=True).copy()
    g["label"] = ""
    ts = g["timestamp"].to_numpy()
    ok = (~ambiguous).to_numpy() & g["lag_1"].notna().to_numpy()
    usable_drop = ok & (g["base_med"].to_numpy() >= DROP_MIN_NORMAL)
    std = g["kwh"].std()
    taken = set()
    kwh = g["kwh"].to_numpy(dtype="float64").copy()
    for i in pick_positions(ok, ts, N_SPIKE, 1, taken, rng):
        kwh[i] += rng.uniform(3, 6) * std
        g.loc[i, "label"] = "spike"
    for i in pick_positions(usable_drop, ts, N_DROP, 1, taken, rng):
        kwh[i] = 0.0
        g.loc[i, "label"] = "drop"
    for i in pick_positions(ok, ts, N_SHIFT_BLOCKS, BLOCK, taken, rng):
        kwh[i:i + BLOCK] *= 2.5
        g.loc[i:i + BLOCK - 1, "label"] = "shift"
    for i in pick_positions(ok, ts, N_FLAT_BLOCKS, BLOCK, taken, rng):
        kwh[i:i + BLOCK] = kwh[i]
        g.loc[i:i + BLOCK - 1, "label"] = "flatline"
    g["kwh"] = kwh.astype("float32")
    return g


def prf(flag, truth):
    tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def injection_test(features, real):
    """Run the injection experiment and return averaged precision / recall / F1 per method and type."""
    houses = (real.groupby("household_id")["city"].first().astype(str).reset_index()
              .groupby("city", group_keys=False).sample(2, random_state=RANDOM_STATE)["household_id"].tolist())
    print(f"Injection test households: {sorted(houses)}")
    amb = real.loc[real["consensus"] >= 2, ["household_id", "timestamp"]].assign(amb=True)
    rows = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(RANDOM_STATE + rep)
        parts = []
        sub = features[features["household_id"].isin(houses)].drop(columns=HISTORY_COLS)
        picked = base_mod.add_baselines(feat_mod.add_history_features(sub))
        picked = picked.merge(amb, on=["household_id", "timestamp"], how="left")
        picked["amb"] = picked["amb"].fillna(False).astype(bool)
        for hid, g in picked.groupby("household_id"):
            parts.append(inject(g, g["amb"], rng))
        inj = pd.concat(parts, ignore_index=True)
        labels = inj[["household_id", "timestamp", "label", "amb"]]
        # recompute everything from the injected kWh, exactly like the real pipeline
        cols = [c for c in features.columns if c not in HISTORY_COLS]
        again = feat_mod.add_history_features(inj[cols])
        again = base_mod.add_baselines(again)
        res = det_mod.detect(again)
        res = res.merge(labels, on=["household_id", "timestamp"], how="left")
        res["flag_cons2"] = res["consensus"] >= 2
        scored = res[~res["amb"]]
        for method, col in METHOD_COLS.items():
            for typ in ["all"] + TYPES:
                keep = (scored["label"] == "") | ((scored["label"] != "") if typ == "all" else (scored["label"] == typ))
                s = scored[keep]
                truth = (s["label"] != "").to_numpy()
                p, r, f = prf(s[col].to_numpy(), truth)
                auc = roc_auc_score(truth, s[SCORE_COLS[method]].to_numpy())
                rows.append(dict(rep=rep, method=method, type=typ, precision=p, recall=r, f1=f, auc=auc, n_injected=int(truth.sum())))
            # block events: was at least one hour of the block flagged?
            for typ in ["shift", "flatline"]:
                s = scored[scored["label"] == typ].copy()
                s["block"] = (s["timestamp"].diff().ne(pd.Timedelta(hours=1)) | s["household_id"].diff().ne(0)).cumsum()
                ev = s.groupby("block")[col].any().mean()
                rows.append(dict(rep=rep, method=method, type=typ + "_event", precision=np.nan, recall=ev, f1=np.nan, n_injected=s["block"].nunique()))
    out = pd.DataFrame(rows).groupby(["method", "type"], sort=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), auc=("auc", "mean"), n_injected=("n_injected", "mean")).round(3).reset_index()
    return out


def rate_tables(real):
    flags = ["flag_if", "flag_lof", "flag_z", "flag_iqr", "flag_global"]
    real = real.assign(city=real["city"].astype(str))
    by_house = real.groupby("household_id")[flags + ["consensus"]].mean()
    by_house["multi_method"] = real.groupby("household_id")["consensus"].apply(lambda s: (s >= 2).mean())
    by_house = by_house.drop(columns="consensus").reset_index()
    by_city = real.groupby("city")[flags].mean().reset_index()
    by_city["multi_method"] = real.groupby("city")["consensus"].apply(lambda s: (s >= 2).mean()).values
    by_month = real.groupby("month")[flags].mean().reset_index()
    by_month["multi_method"] = real.groupby("month")["consensus"].apply(lambda s: (s >= 2).mean()).values
    return by_house, by_city, by_month


def agreement(real):
    cols = ["flag_if", "flag_lof", "flag_z", "flag_iqr"]
    rows = []
    for a in cols:
        for b in cols:
            inter = (real[a] & real[b]).sum(); union = (real[a] | real[b]).sum()
            rows.append(dict(method_a=a, method_b=b, jaccard=round(inter / union, 3)))
    return pd.DataFrame(rows)


def pick_cases(real):
    """5 representative flagged cases + 2 that look like false alarms. Different homes each time."""
    cases = []
    strong = real[real["consensus"] == 4].sort_values("if_score", ascending=False)
    used = set()
    for reason in strong["reason"].astype(str).unique():
        for _, r in strong[strong["reason"].astype(str) == reason].iterrows():
            if r["household_id"] not in used:
                cases.append((r, "Strong flag (all 4 methods agree)")); used.add(r["household_id"]); break
        if len(cases) == 5:
            break
    for _, r in strong.iterrows():
        if len(cases) == 5:
            break
        if r["household_id"] not in used:
            cases.append((r, "Strong flag (all 4 methods agree)")); used.add(r["household_id"])
    weak = real[(real["consensus"] == 1) & real["flag_if"] & (real["weather_z"].abs() < Z_THRESHOLD)].sort_values("if_score", ascending=False)
    fa = 0
    for _, r in weak.iterrows():
        if r["household_id"] not in used:
            cases.append((r, "Possible false alarm (only 1 of 4 methods flags it)")); used.add(r["household_id"]); fa += 1
        if fa == 2:
            break
    rows = []
    for i, (r, kind) in enumerate(cases, 1):
        note = (f"{r['reason']}. Usage {r['kwh']:.2f} kWh vs normal {r['base_med']:.2f} kWh for this hour "
                f"(robust score {r['robust_z']:.1f}, weather-adjusted score {r['weather_z']:.1f}, temperature {r['temp_c']:.0f} C).")
        rows.append(dict(case_id=i, kind=kind, household_id=int(r["household_id"]), city=str(r["city"]),
                         timestamp=r["timestamp"], kwh=round(float(r["kwh"]), 3), normal_kwh=round(float(r["base_med"]), 3),
                         consensus=int(r["consensus"]), note=note))
    return pd.DataFrame(rows)


def weather_check(real):
    """Do flags cluster in hot hours? Is the weather-aware score less fooled by heat?"""
    r = real.copy()
    r["hot"] = r["temp_c"] >= r.groupby("household_id")["temp_c"].transform(lambda s: s.quantile(0.9))
    hot, rest = r[r["hot"]], r[~r["hot"]]
    rows = []
    for name, col in [("Isolation Forest flag", "flag_if"), ("Z-score flag", "flag_z"), ("IQR flag", "flag_iqr"), ("Global z-score flag", "flag_global")]:
        rows.append(dict(measure=name, hot_hours_pct=100 * hot[col].mean(), other_hours_pct=100 * rest[col].mean()))
    high_time = r["robust_z"] > Z_THRESHOLD
    high_weather = r["weather_z"] > Z_THRESHOLD
    rows.append(dict(measure="High vs time-only baseline (robust score > 3.5)", hot_hours_pct=100 * high_time[r["hot"]].mean(), other_hours_pct=100 * high_time[~r["hot"]].mean()))
    rows.append(dict(measure="High vs weather-aware baseline (score > 3.5)", hot_hours_pct=100 * high_weather[r["hot"]].mean(), other_hours_pct=100 * high_weather[~r["hot"]].mean()))
    out = pd.DataFrame(rows).round(2)
    out["ratio_hot_vs_other"] = (out["hot_hours_pct"] / out["other_hours_pct"]).round(2)
    return out


def main():
    ensure_dirs()
    real = pd.read_parquet(PROCESSED / "anomalies_full.parquet")
    features = pd.read_parquet(PROCESSED / "features.parquet")

    print("=== 1. Injection test ===")
    inj = injection_test(features, real)
    inj.to_csv(APP_DATA / "injection_results.csv", index=False)
    print(inj[inj.type == "all"].to_string(index=False))
    print("Recall by type:")
    print(inj[inj.type.isin(TYPES + ["shift_event", "flatline_event"])].pivot(index="method", columns="type", values="recall").to_string())
    print("AUC by type:")
    print(inj[inj.type.isin(TYPES)].pivot(index="method", columns="type", values="auc").to_string())

    print("\n=== 2. Anomaly rates ===")
    by_house, by_city, by_month = rate_tables(real)
    by_house.to_csv(APP_DATA / "rate_by_household.csv", index=False)
    by_city.to_csv(APP_DATA / "rate_by_city.csv", index=False)
    by_month.to_csv(APP_DATA / "rate_by_month.csv", index=False)
    print((by_city.set_index("city") * 100).round(2).to_string())

    print("\n=== 3. Agreement (Jaccard) ===")
    agr = agreement(real)
    agr.to_csv(APP_DATA / "agreement.csv", index=False)
    print(agr.pivot(index="method_a", columns="method_b", values="jaccard").to_string())

    print("\n=== 4. Cases ===")
    cases = pick_cases(real)
    cases.to_csv(APP_DATA / "cases.csv", index=False)
    ctx = []
    for _, c in cases.iterrows():
        w = real[(real.household_id == c.household_id) & (real.timestamp.between(c.timestamp - pd.Timedelta(hours=72), c.timestamp + pd.Timedelta(hours=72)))]
        ctx.append(w.assign(case_id=c.case_id)[["case_id", "household_id", "timestamp", "kwh", "base_med", "temp_c", "flag_if", "consensus"]])
    pd.concat(ctx).to_parquet(APP_DATA / "case_context.parquet", index=False)
    print(cases[["case_id", "kind", "household_id", "city", "timestamp", "note"]].to_string(index=False))

    print("\n=== 5. Weather check ===")
    wc = weather_check(real)
    wc.to_csv(APP_DATA / "weather_check.csv", index=False)
    print(wc.to_string(index=False))

    # method comparison table
    overall = inj[inj.type == "all"].set_index("method")
    comp = pd.DataFrame({"flag_rate_pct": {m: round(100 * (real["consensus"].ge(2).mean() if c == "flag_cons2" else real[c].mean()), 2) for m, c in METHOD_COLS.items()}})
    comp = comp.join(overall[["precision", "recall", "f1", "auc"]]).reset_index().rename(columns={"index": "method"})
    comp.to_csv(APP_DATA / "method_comparison.csv", index=False)
    print("\n=== Method comparison ===")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
