"""Run the whole pipeline in order:  python run_pipeline.py

Needs the raw REWD-P files in data/raw/ (see README). Each step prints its own checks.
Use --from N to restart at step N (e.g. --from 5 skips the slow data loading).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent / "src"
STEPS = ["01_load_and_clean.py", "02_integrate_weather.py", "03_features.py", "04_baseline_profiles.py",
         "04b_eda_data.py", "05_anomaly_detection.py", "06_clustering.py", "07_evaluation.py", "08_make_report.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step number to run (1-8)")
    start = ap.parse_args().start
    for i, step in enumerate(STEPS, start=1):
        if int(step[:2]) < start:
            continue
        print(f"\n{'=' * 70}\nRunning {step}\n{'=' * 70}", flush=True)
        t0 = time.time()
        # cwd=src so the scripts can `import utils`
        result = subprocess.run([sys.executable, step], cwd=SRC)
        if result.returncode != 0:
            sys.exit(f"Step {step} failed. Fix it and rerun (use --from {step[:2]} to resume).")
        print(f"[{step} done in {time.time() - t0:.0f}s]")
    print("\nPipeline finished. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
