"""Step 1 + 2: load the raw REWD-P files, inspect them, clean them, make HOURLY data.

Why: the raw data is 1-minute readings (~30 million rows). We never load it all at
once. We read ONE house at a time, keep only the whole-house column, clean it,
resample to hourly and save one small Parquet file.

Facts verified from the real files (see docs/research_notes.md):
  * the whole-house column is "Usage (kW)" (spelled slightly differently in a few files)
  * values are POWER in kW, so the hourly MEAN kW equals the energy in kWh for that hour
  * timestamps at midnight have no time part ("2023-11-01")
  * Peshawar files exist twice (House21.csv and peshawar_House21.csv, identical)

Cleaning rules (every rule is counted and written to cleaning_log.csv):
  1. unreadable timestamp  -> row dropped
  2. duplicate timestamp   -> readings with the same timestamp are averaged
  3. negative or > MAX_KW  -> set to missing (physically invalid)
  4. high-but-valid values -> KEPT (a hot spell can legitimately be high)
  5. zeros                 -> KEPT (can be a real outage, which is interesting)
  6. an hour is valid only if it has at least MIN_MINUTES valid minutes

Timezone: the dataset does not state one. We assume local Pakistan time and keep
timestamps "naive" (no timezone attached).
"""
import re
import numpy as np
import pandas as pd

from utils import RAW, PROCESSED, ensure_dirs

MAX_KW = 50.0       # a home meter cannot read more than this; nothing in the data reaches it
MIN_MINUTES = 45    # at least 45 of 60 minutes must be present for a valid hour

ENERGY_DIR = RAW / "energy" / "rewdp_dataset"
WEATHER_DIR = RAW / "weather" / "weather_dataset"
META_FILE = RAW / "metadata_ultimate.xlsx"


def find_energy_files():
    """Return {house_id: (city, path)}. Prefer '<city>_HouseN.csv' over duplicate 'HouseN.csv'."""
    files = {}
    for city_dir in sorted(p for p in ENERGY_DIR.iterdir() if p.is_dir()):
        for f in sorted(city_dir.glob("*.csv")):
            house = int(re.search(r"House(\d+)", f.name).group(1))
            has_prefix = f.name.lower().startswith(city_dir.name.lower())
            if house not in files or has_prefix:
                files[house] = (city_dir.name, f)
    return files


def parse_time(series):
    """Parse timestamps. Rows at midnight have only a date, so try both formats."""
    full = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    date_only = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    return full.fillna(date_only)


def clean_one_house(house, city, path):
    """Read, clean and resample one household. Returns (hourly_df, log_dict, inventory_dict)."""
    header = pd.read_csv(path, nrows=0).columns
    usage_col = [c for c in header if c.lower().startswith("usage")][0]
    raw = pd.read_csv(path, usecols=["datetime", usage_col], dtype={usage_col: "float32"})
    n_rows = len(raw)
    n_appliance_cols = len(header) - 2

    t = parse_time(raw["datetime"])
    bad_time = int(t.isna().sum())
    df = pd.DataFrame({"t": t, "kw": raw[usage_col].astype("float32")}).dropna(subset=["t"])

    n_missing_raw = int(df["kw"].isna().sum())
    invalid = (df["kw"] < 0) | (df["kw"] > MAX_KW)
    n_invalid = int(invalid.sum())
    df.loc[invalid, "kw"] = np.nan

    dup_ts = int(df["t"].duplicated().sum())
    df = df.groupby("t", sort=True)["kw"].mean()          # sorted + duplicates averaged

    hourly = df.resample("h").agg(["mean", "count"])
    hourly.columns = ["kwh", "n_minutes"]
    hours_in_span = len(hourly)
    valid = hourly["n_minutes"] >= MIN_MINUTES
    n_short_hours = int((~valid).sum())
    hourly = hourly[valid].copy()
    hourly["kwh"] = hourly["kwh"].astype("float32")
    hourly["n_minutes"] = hourly["n_minutes"].astype("int8")
    hourly.index.name = "timestamp"
    hourly = hourly.reset_index()
    hourly.insert(0, "city", pd.Categorical([city] * len(hourly)))
    hourly.insert(0, "household_id", np.int16(house))

    log = dict(household_id=house, city=city, raw_rows=n_rows, bad_timestamp_rows=bad_time,
               duplicate_timestamps_merged=dup_ts, missing_values_in_raw=n_missing_raw,
               invalid_values_set_missing=n_invalid, hours_in_span=hours_in_span,
               hours_dropped_too_few_minutes=n_short_hours, hours_kept=len(hourly))
    inv = dict(household_id=house, city=city, file=path.name, usage_column=usage_col,
               appliance_columns=n_appliance_cols, first_time=hourly["timestamp"].min(),
               last_time=hourly["timestamp"].max(),
               missing_hours_pct=round(100 * n_short_hours / hours_in_span, 2),
               mean_kwh=round(float(hourly["kwh"].mean()), 3),
               max_kwh=round(float(hourly["kwh"].max()), 2),
               zero_hours_pct=round(100 * float((hourly["kwh"] == 0).mean()), 2))
    return hourly, log, inv


def load_metadata():
    """Clean the metadata sheet: tidy column names, numeric house id."""
    meta = pd.read_excel(META_FILE)
    meta.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in meta.columns]
    meta["household_id"] = meta["House"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return meta


def main():
    ensure_dirs()
    files = find_energy_files()
    print(f"Found {len(files)} unique households in {ENERGY_DIR}")
    parts, logs, invs = [], [], []
    for house in sorted(files):
        city, path = files[house]
        hourly, log, inv = clean_one_house(house, city, path)
        parts.append(hourly); logs.append(log); invs.append(inv)
        print(f"  House {house:>2} ({city:<9}) raw rows {log['raw_rows']:>7,} -> valid hours {log['hours_kept']:>5,}")

    hourly = pd.concat(parts, ignore_index=True)
    log = pd.DataFrame(logs)
    inv = pd.DataFrame(invs)
    hourly.to_parquet(PROCESSED / "hourly_energy.parquet", index=False)
    log.to_csv(PROCESSED / "cleaning_log.csv", index=False)
    inv.to_csv(PROCESSED / "data_inventory.csv", index=False)
    meta = load_metadata()
    meta.to_csv(PROCESSED / "metadata_clean.csv", index=False)

    # Sanity checks and verification of the proposal's claims
    print("\n=== Verification of proposal claims ===")
    print(f"Households found       : {inv.household_id.nunique()} (proposal says 60)")
    print(f"Cities                 : {sorted(inv.city.unique())}")
    print(f"Energy time range      : {hourly.timestamp.min()} -> {hourly.timestamp.max()}")
    wfirst = pd.read_csv(WEATHER_DIR / "Lahore.csv", usecols=["datetime"])["datetime"]
    print(f"Weather time range     : {wfirst.iloc[0]} -> {wfirst.iloc[-1]} (hourly)")
    print(f"Metadata rows          : {len(meta)}")
    print("\n=== Totals ===")
    print(f"Raw 1-minute rows      : {log.raw_rows.sum():,}")
    print(f"Duplicate timestamps   : {log.duplicate_timestamps_merged.sum():,} merged")
    print(f"Invalid values removed : {log.invalid_values_set_missing.sum():,}")
    print(f"Hourly rows kept       : {len(hourly):,}")
    print(f"NaN / inf in kwh       : {int(hourly.kwh.isna().sum())} / {int(np.isinf(hourly.kwh).sum())}")
    assert hourly.duplicated(["household_id", "timestamp"]).sum() == 0
    assert set(inv.household_id) == set(meta.household_id), "metadata and energy houses differ"
    print("Checks passed.")


if __name__ == "__main__":
    main()
