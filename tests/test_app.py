"""Opens every page of the Streamlit app, changes filters, and fails on any exception.

Run:  python tests/test_app.py        (from the project root, with the .venv active)
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["Home", "Data Explorer", "Normal Behaviour", "Anomaly Detector", "Household Clusters",
         "Weather Impact", "Method Comparison", "About & Limitations"]


def fresh(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


def check(at, label):
    errors = [e.value for e in at.exception]
    assert not errors, f"{label}: {errors}"
    print(f"OK  {label}")


def main():
    for page in PAGES:
        check(fresh(page), f"{page} (defaults)")

    # Data Explorer: city + household filters
    at = fresh("Data Explorer")
    at.selectbox[0].select("Skardu").run(); check(at, "Data Explorer: city = Skardu")
    at.selectbox[1].select_index(1).run(); check(at, "Data Explorer: first household")

    # Normal Behaviour: a few households
    at = fresh("Normal Behaviour")
    for i in (0, 25, 58):
        at.selectbox[0].select_index(i).run(); check(at, f"Normal Behaviour: household #{i}")

    # Anomaly Detector: every method on a few households
    at = fresh("Anomaly Detector")
    for i in (0, 30, 58):
        at.selectbox[0].select_index(i).run()
        for m in at.selectbox[1].options:
            at.selectbox[1].select(m).run(); check(at, f"Anomaly Detector: household #{i}, {m}")

    # Weather Impact and Method Comparison: switch inner controls
    at = fresh("Weather Impact")
    at.selectbox[0].select_index(10).run(); check(at, "Weather Impact: household #10")
    at = fresh("Method Comparison")
    for m in ("Recall (found)", "F1 (balance)", "AUC (ranking quality)"):
        at.main.radio[0].set_value(m).run(); check(at, f"Method Comparison: metric {m}")
    # Navigation buttons: the Home call-to-action and the Next / Back footer
    at = fresh("Home")
    [b for b in at.button if b.key == "cta_home"][0].click().run()
    assert at.sidebar.radio[0].value == "Anomaly Detector", "call-to-action did not open the Anomaly Detector"
    check(at, "Home: call-to-action opens Anomaly Detector")
    [b for b in at.button if b.key.startswith("next_")][0].click().run()
    assert at.sidebar.radio[0].value == "Household Clusters", "Next button did not advance"
    check(at, "Footer: Next opens Household Clusters")
    [b for b in at.button if b.key.startswith("back_")][0].click().run()
    assert at.sidebar.radio[0].value == "Anomaly Detector", "Back button did not go back"
    check(at, "Footer: Back returns to Anomaly Detector")

    # Deep link: ?page=Weather Impact opens that page directly
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.query_params["page"] = "Weather Impact"
    at.run()
    assert at.sidebar.radio[0].value == "Weather Impact", "?page= deep link did not open the page"
    check(at, "Deep link: ?page=Weather Impact")

    print("\nAll app checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        sys.exit(f"FAILED: {e}")
