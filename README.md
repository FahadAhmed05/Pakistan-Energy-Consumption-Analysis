# Pakistan Electricity Consumption Anomaly Mining

Data Mining course project, 2026.
**Team:** Izaan Imran, Fahad Ahmed, Shiraz Asif, Dayyan Hashmi

We learn what is *normal electricity use for each home* and flag the hours that differ from it, using real
one-minute smart-meter data from Pakistani households and the weather of each city. Results are shown in a
Streamlit app. The full write-up is in [docs/Project_Report.pdf](docs/Project_Report.pdf).

> **An anomaly flag only means "differs from the learned normal pattern".** It is not proof of electricity
> theft, equipment failure or any other specific cause.

## Data

Residential Energy and Weather Data Pakistan (REWD-P), LUMS Energy Institute:
<https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html>

What we found in the real files (this differs slightly from the proposal):

| Item | Proposal | Real files |
|---|---|---|
| Households | 60 | **59** (Skardu has 9) |
| Energy resolution | 1 minute | 1 minute, power in kW |
| Energy period | 1 year | about 1 year per home; Jul 2023 to Nov 2024 overall |
| Weather | 18 months | hourly, Jul 2023 to Nov 2024, 6 cities |

## Folder structure

```
app.py                  Streamlit app (reads app_data/ only)
run_pipeline.py         runs every step in order
src/                    01 load+clean, 02 weather join, 03 features, 04 baselines, 04b EDA tables,
                        05 anomaly detection, 06 clustering, 07 evaluation, 08 PDF report, utils.py
tests/test_app.py       opens every app page and fails on any exception
app_data/               small result files used by the app (committed, about 16 MB)
data/processed/         cleaned hourly data and logs
data/raw/               original files (NOT committed, see below)
docs/                   research_notes.md and Project_Report.pdf
requirements.txt        packages for the app (what Streamlit Cloud installs)
requirements-pipeline.txt  packages to rebuild everything
```

## Setup

Python 3.13 was used (3.12 or newer should work).

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements-pipeline.txt     # use requirements.txt if you only want the app
```

## Run the app

The repository already contains the precomputed results, so this works right after setup:

```bash
streamlit run app.py
```

## Rebuild everything from the raw data

1. Download these three files from the dataset page above into `data/raw/`:
   - `rewdp_dataset.zip` (about 505 MB)
   - `weather_dataset.zip`
   - `metadata_ultimate.xlsx`
   Direct links: `https://web.lums.edu.pk/~eig/gallery/datasets/rewdp/<file name>`
2. Unzip them so the folders look like this:
   - `data/raw/energy/rewdp_dataset/<City>/*.csv`
   - `data/raw/weather/weather_dataset/*.csv`
   - `data/raw/metadata_ultimate.xlsx`
3. Run:

```bash
python run_pipeline.py            # about 8 minutes on a laptop
python run_pipeline.py --from 5   # restart from step 5 if something fails later
python tests/test_app.py          # checks every app page
```

The raw data is processed one household at a time and reduced to hourly Parquet files, so memory stays low
(the raw files use about 2.6 GB of disk).

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (raw data is ignored by `.gitignore`).
2. On <https://share.streamlit.io> choose **New app** and pick the repository.
3. Branch `main`, main file path `app.py`.
4. Under **Advanced settings** choose Python **3.13**.

The app only reads files from `app_data/` (about 16 MB) and never trains a model, so it fits the free tier.

## Method in short

1. Clean minute data to **hourly kWh** (the hourly mean of kW equals kWh); keep real highs and zeros.
2. Join hourly weather of the home's city.
3. Features: hour, day, month, peak flag, lags, rolling stats, change, flat-run, weather (past values only).
4. **Normal pattern** per home: median and MAD by hour x weekday/weekend, and by hour x temperature band.
5. **Detection** per home: Isolation Forest (main), Local Outlier Factor, Z-score, IQR, plus a naive global
   z-score for comparison; consensus count and a plain-language reason for every flag; daily-window anomalies.
6. **Clustering:** K-Means on household behaviour, drawn with PCA.
7. **Evaluation without labels:** injected anomalies (precision, recall, F1, AUC), agreement, rates, example cases,
   weather check.

## Limitations

- No real anomaly labels, so accuracy is measured with injected (synthetic) anomalies.
- 59 households only; coverage differs per home and some have gaps.
- Timezone is not stated in the data; local Pakistan time is assumed.
- Normal behaviour is learned from the whole recorded period (offline analysis).
- Cluster separation is modest (silhouette about 0.2).
