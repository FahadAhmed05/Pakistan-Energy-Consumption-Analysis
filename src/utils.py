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

# Peak hours (evening): chosen from the EDA in step 4, see docs/research_notes.md
PEAK_HOURS = list(range(18, 23))

# Colours used in every chart and the app (colour-blind safe blue vs orange)
COLOR_NORMAL = "#1f6f8b"
COLOR_ANOMALY = "#e8710a"


def ensure_dirs():
    """Create output folders if they do not exist yet."""
    for p in (PROCESSED, APP_DATA, DOCS):
        p.mkdir(parents=True, exist_ok=True)
