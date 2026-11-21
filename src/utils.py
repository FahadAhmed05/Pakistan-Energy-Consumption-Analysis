"""Shared paths, constants and small helpers used by every pipeline step.

Why: keeping paths and settings in one place means no script hard-codes a
folder, and the whole project runs the same on Windows, Mac and Linux.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
APP_DATA = ROOT / "app_data"
DOCS = ROOT / "docs"

RANDOM_STATE = 42

# Peak hours: the 5 hours with the highest average load across all homes (EDA, step 4).
# Late evening / night, most likely cooling and evening activity.
PEAK_HOURS = [21, 22, 23, 0, 1]

# Robust z-scores beyond this are called unusual (common cut-off for modified z-scores)
Z_THRESHOLD = 3.5

# Colours used in every chart and the app (colour-blind safe blue vs orange)
COLOR_NORMAL = "#2a78d6"
COLOR_ANOMALY = "#eb6834"


def ensure_dirs():
    """Create output folders if they do not exist yet."""
    for p in (PROCESSED, APP_DATA, DOCS):
        p.mkdir(parents=True, exist_ok=True)
