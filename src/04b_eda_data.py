"""Step 4: exploratory data analysis. Saves small tables that the app and the report plot.

Outputs (app_data/):
  profile_hourly.csv   per household x weekday/weekend x hour: normal level and band
  daily_series.parquet per household x day: total kWh and mean temperature
  city_month.csv       average hourly kWh per city and month
  city_hour.csv        average 24-hour profile per city (used to check the peak hours)
  temp_response.csv    per household x temperature band (2 C): average kWh
  weather_corr.csv     per household: correlation of kWh with temperature / humidity / rain
  household_summary.csv per household: level, peak hour, coverage
"""
import numpy as np
import pandas as pd

from utils import PROCESSED, APP_DATA, PEAK_HOURS, ensure_dirs


def q(p):
    return lambda s: s.quantile(p)


def main():
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "baselined.parquet")
    df["city"] = df["city"].astype(str)

    prof = df.groupby(["household_id", "is_weekend", "hour"])["kwh"].agg(
        normal="median", mean="mean", p10=q(0.10), p90=q(0.90)).round(4).reset_index()
    prof.to_csv(APP_DATA / "profile_hourly.csv", index=False)

    d = df.assign(date=df["timestamp"].dt.floor("D"))
    daily = d.groupby(["household_id", "city", "date"]).agg(
        hours=("kwh", "size"), total_kwh=("kwh", "sum"), mean_temp=("temp_c", "mean")).reset_index()
    daily = daily[daily.hours >= 20]
    daily["total_kwh"] = (daily["total_kwh"] / daily["hours"] * 24).astype("float32")
    daily["mean_temp"] = daily["mean_temp"].astype("float32")
    daily.drop(columns="hours").to_parquet(APP_DATA / "daily_series.parquet", index=False)

    df.groupby(["city", "month"])["kwh"].mean().round(4).reset_index().to_csv(APP_DATA / "city_month.csv", index=False)
    city_hour = df.groupby(["city", "hour"])["kwh"].mean().round(4).reset_index()
    city_hour.to_csv(APP_DATA / "city_hour.csv", index=False)

    df["temp_bin"] = (np.floor(df["temp_c"] / 2) * 2).astype(int)
    tr = df.groupby(["household_id", "city", "temp_bin"])["kwh"].agg(["mean", "size"]).reset_index()
    tr[tr["size"] >= 20].rename(columns={"mean": "mean_kwh", "size": "hours"}).round(4).to_csv(APP_DATA / "temp_response.csv", index=False)

    corr = df.groupby("household_id").apply(lambda g: pd.Series({
        "temp": g["kwh"].corr(g["temp_c"]), "humidity": g["kwh"].corr(g["humidity"]),
        "precip": g["kwh"].corr(g["precip_mm"])}), include_groups=False).round(3).reset_index()
    corr.insert(1, "city", df.groupby("household_id")["city"].first().values)
    corr.to_csv(APP_DATA / "weather_corr.csv", index=False)

    summ = df.groupby(["household_id", "city"]).agg(hours=("kwh", "size"), mean_kwh=("kwh", "mean"),
              max_kwh=("kwh", "max"), first=("timestamp", "min"), last=("timestamp", "max")).reset_index()
    summ[["mean_kwh", "max_kwh"]] = summ[["mean_kwh", "max_kwh"]].round(3)
    summ["peak_hour"] = summ["household_id"].map(df.groupby(["household_id", "hour"])["kwh"].mean().groupby("household_id").idxmax().map(lambda t: t[1]))
    summ.to_csv(APP_DATA / "household_summary.csv", index=False)

    print("Average kWh by hour (all homes):")
    overall = df.groupby("hour")["kwh"].mean()
    print(overall.round(3).to_string())
    top5 = overall.sort_values(ascending=False).index[:5].tolist()
    print(f"\nTop-5 hours in the data: {sorted(top5)}   assumed peak hours: {PEAK_HOURS}")
    print("\nPeak hour per city:", city_hour.loc[city_hour.groupby("city")["kwh"].idxmax()].set_index("city")["hour"].to_dict())
    print("\nMean correlation of kWh with weather (per household):")
    print(corr[["temp", "humidity", "precip"]].mean().round(3).to_string())
    print("\nAverage kWh by city:")
    print(df.groupby("city")["kwh"].mean().round(3).to_string())
    print("\nAverage kWh by month (all homes):")
    print(df.groupby("month")["kwh"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
