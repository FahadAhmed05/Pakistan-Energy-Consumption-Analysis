"""Step 3: join hourly electricity with hourly weather (same city, same hour).

Why: the proposal wants to separate "high because it is hot" from "truly unusual",
so every electricity row needs the weather of its own city at that hour.

Weather cleaning: timestamps parsed (midnight rows have no time part), duplicates
dropped, short gaps (up to 3 hours) filled by interpolation, values checked.
"""
import pandas as pd

from utils import RAW, PROCESSED

WEATHER_DIR = RAW / "weather" / "weather_dataset"
KEEP = ["Temperature", "Humidity", "Precipitation", "Wind Speed", "Pressure", "Solar Radiation"]
RENAME = {"Temperature": "temp_c", "Humidity": "humidity", "Precipitation": "precip_mm",
          "Wind Speed": "wind_speed", "Pressure": "pressure", "Solar Radiation": "solar_rad"}


def load_weather():
    parts = []
    for f in sorted(WEATHER_DIR.glob("*.csv")):
        w = pd.read_csv(f, usecols=["datetime"] + KEEP)
        t = pd.to_datetime(w["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        w["timestamp"] = t.fillna(pd.to_datetime(w["datetime"], format="%Y-%m-%d", errors="coerce"))
        n_bad = int(w["timestamp"].isna().sum())
        w = w.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
        w = w[KEEP].resample("h").mean()
        n_gap = int(w["Temperature"].isna().sum())
        w = w.interpolate(limit=3)
        print(f"  {f.stem:<10} rows {len(w):,}  bad timestamps {n_bad}  gap hours {n_gap}  still missing {int(w.isna().any(axis=1).sum())}")
        w = w.rename(columns=RENAME).reset_index()
        w.insert(0, "city", f.stem)
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def main():
    energy = pd.read_parquet(PROCESSED / "hourly_energy.parquet")
    energy["city"] = energy["city"].astype(str)
    weather = load_weather()
    merged = energy.merge(weather, on=["city", "timestamp"], how="left")

    unmatched = int(merged["temp_c"].isna().sum())
    print(f"\nEnergy rows: {len(energy):,}  merged rows: {len(merged):,}  without weather: {unmatched:,}")
    assert len(merged) == len(energy), "join changed the row count"
    merged = merged.dropna(subset=["temp_c", "humidity", "precip_mm"]).reset_index(drop=True)
    print(f"Rows kept after dropping missing weather: {len(merged):,}")
    for c in ["temp_c", "humidity", "precip_mm"]:
        print(f"  {c}: min {merged[c].min():.1f}  max {merged[c].max():.1f}")
    merged["city"] = merged["city"].astype("category")
    for c in ["temp_c", "humidity", "precip_mm", "wind_speed", "pressure", "solar_rad"]:
        merged[c] = merged[c].astype("float32")
    merged.to_parquet(PROCESSED / "hourly_merged.parquet", index=False)


if __name__ == "__main__":
    main()
