# Research notes

Only sources that were actually opened are listed.

## 1. REWD-P dataset (source: LUMS Energy Institute page)
URL: https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html

What the page says:
- One year of smart-meter data, 1-minute resolution, from households in six cities
  (Lahore, Multan, Peshawar, Karachi, Islamabad, Skardu). Weather for 18 months.
- Files are public direct downloads: one energy ZIP, one weather ZIP, one metadata XLSX.
- The file list has **59** houses (10 each, except Skardu with 9), although the text and our
  proposal say 60. We report 59 and note the difference.

What we found by opening the real files:
- Metadata: `metadata_ultimate.xlsx`, 59 rows, 46 columns (city, people, rooms, appliance counts...).
- Weather: one hourly CSV per city, 12,456 rows each, starting 2023-07-01, 10 variables
  (Temperature, Humidity, Dew, Precipitation, Wind Speed/Direction, Pressure, Solar Radiation/Energy, UV Index).
- Energy: columns `datetime`, `Usage (kW)` plus a few appliance columns. Values are **power in kW**,
  averaged per minute. Midnight timestamps have no time part (`2023-11-01`), so parsing must accept mixed formats.
  Energy starts 2023-11-01, so it overlaps the weather period.
- No timezone or documentation was published. We assume local Pakistan time (PKT) and keep timestamps naive.

## 2. Design decisions this led to
| Finding | Decision |
|---|---|
| Values are kW (power) | Hourly **mean kW = kWh in that hour**. We label axes "kWh per hour". |
| ~525k rows per house | Read one file at a time, keep only needed columns, resample to hourly, save Parquet. |
| Weather is hourly | Join on (city, hour). No interpolation needed. |
| No anomaly labels | Unsupervised methods + a synthetic injection test for validation. |
| Households differ a lot in size | Baselines and models are per household, using robust median/MAD. |
| Weather drives cooling load | Weather-adjusted baseline (temperature bins) to separate "hot" from "unusual". |

## 3. Standard practice (general knowledge, not a cited source)
Common unsupervised smart-meter approaches are statistical thresholds (z-score, IQR), Isolation Forest, LOF,
and clustering of load profiles. Robust statistics (median/MAD) are preferred over mean/std because
outliers inflate the std. We follow these ideas; we do not cite specific papers.

## 4. UI/UX research (app redesign)
Sources actually opened or searched (summaries read; several pages refused automated access, so they are not relied on):
- Toptal, UXPilot and the Streamlit dashboard docs were read: lead with 3 to 5 headline KPIs, summary then trends then detail,
  progressive disclosure, one global filter row, the "hover test" (a chart must make sense without hovering), do not rely on
  red/green alone, plain-language empty states.
- Search results on compact density (Material, Cloudscape, Microsoft, Atlassian spacing pages): a 4px base grid, 8px compact
  component padding, 12 to 16px card padding, 12 to 14px text; Material warns that density hurts accessibility if pushed too far,
  so controls stay at least about 32px tall and horizontal margins stay generous.
- The dataviz palette validator checked every colour set (normal/flagged pair and the cluster trio, light and dark).

Decisions: compact layout with graphs left at full size; blue/orange pair; insight titles; table view under charts;
Back / Next footer for story-order presenting; `?page=` deep links; native Streamlit light and dark themes
(Streamlit exposes no CSS variables, so a custom toggle could not restyle its own widgets and dataframes).
