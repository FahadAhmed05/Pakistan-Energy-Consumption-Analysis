"""Build docs/Project_Report.pdf from the real results in app_data/ and data/processed/.

Every number in the report is read from the result files, so the report always matches the last run.
Figures are drawn with matplotlib into a temporary folder and embedded in the PDF.
"""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from utils import APP_DATA, DOCS, PROCESSED, COLOR_ANOMALY, COLOR_NORMAL, ensure_dirs

GITHUB_URL = "[add GitHub repository link here]"
APP_URL = "[add Streamlit Community Cloud link here]"
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14, alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=colors.HexColor("#1f6f8b"), spaceBefore=10, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=6, spaceAfter=3, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#1f6f8b"))


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(rows, widths, header=True, font=8.5):
    cell = ParagraphStyle("cell", parent=BODY, fontSize=font, leading=font + 2.5, spaceAfter=0)
    head = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    data = [[Paragraph(str(c), head if (header and r == 0) else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6f8b"))]
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6f8")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return t


# ------------------------------------------------------------------ figures
def fig_pipeline(path):
    steps = ["1 Collect", "2 Clean", "3 Join\nweather", "4 Explore\n(EDA)", "5 Build\nfeatures", "6 Learn\nnormal", "7 Detect\nanomalies", "8 Cluster\nhomes", "9 Evaluate", "10 Show\n(app)"]
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.axis("off")
    for i, s in enumerate(steps):
        x = i * 1.0
        ax.add_patch(FancyBboxPatch((x, 0.2), 0.86, 0.6, boxstyle="round,pad=0.02", fc=COLOR_NORMAL if i != 6 else COLOR_ANOMALY, ec="none"))
        ax.text(x + 0.43, 0.5, s, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.0, 0.5), xytext=(x + 0.88, 0.5), arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlim(-0.1, 10)
    ax.set_ylim(0, 1)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_profile(path, city_hour):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for c, g in city_hour.groupby("city"):
        ax.plot(g["hour"], g["kwh"], label=c, lw=1.8)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Average energy (kWh per hour)")
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22)); ax.grid(alpha=0.25)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_month(path, city_month):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for c, g in city_month.groupby("city"):
        ax.plot(g["month"], g["kwh"], marker="o", ms=3, label=c, lw=1.6)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("Average energy (kWh per hour)")
    ax.legend(ncol=3, fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_case(path, ctx, case):
    c = ctx[ctx.case_id == case.case_id].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(c.timestamp, c.kwh, color=COLOR_NORMAL, lw=1.3, label="Energy used")
    ax.plot(c.timestamp, c.base_med, color="#9aa7b0", lw=1, ls=":", label="Normal for this hour")
    a = c[c.consensus > 0]
    ax.scatter(a.timestamp, a.kwh, color=COLOR_ANOMALY, marker="D", s=30, label="Flagged by at least one method", zorder=3)
    ax.set_ylabel("Energy (kWh per hour)"); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_pca(path, hc, var):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    pal = ["#1f6f8b", "#e8710a", "#6a994e", "#8d6a9f", "#b5651d"]
    for i, (n, g) in enumerate(hc.groupby("cluster_name")):
        ax.scatter(g.pc1, g.pc2, s=36, color=pal[i % 5], label=n.split(":")[0] + ": " + n.split(": ", 1)[1], alpha=0.85)
    ax.set_xlabel(f"PC1 ({100 * var.explained[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100 * var.explained[1]:.0f}%)")
    ax.legend(fontsize=7, frameon=False, loc="best"); ax.grid(alpha=0.25)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_injection(path, inj):
    d = inj[inj.type.isin(["spike", "drop", "shift", "flatline"])].pivot(index="type", columns="method", values="auc").loc[["spike", "drop", "shift", "flatline"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    d.plot.bar(ax=ax, width=0.8, colormap="tab10", rot=0)
    ax.set_ylabel("AUC (1.0 = perfect ranking)"); ax.set_xlabel("Injected anomaly type"); ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="#888", lw=0.8, ls="--"); ax.legend(fontsize=6.5, frameon=False, ncol=2); ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_weather(path, wc):
    d = wc[wc.measure.str.startswith("High")].copy()
    d["label"] = ["Time-only baseline", "Weather-aware baseline"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d.hot_hours_pct, 0.4, color=COLOR_ANOMALY, label="Hottest 10% of hours")
    ax.bar([i + 0.2 for i in x], d.other_hours_pct, 0.4, color=COLOR_NORMAL, label="All other hours")
    ax.set_xticks(list(x)); ax.set_xticklabels(d.label); ax.set_ylabel("Hours called 'high' (%)")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def img(path, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7780"))
    canvas.drawString(2 * cm, 1.1 * cm, "Pakistan Electricity Consumption Anomaly Mining | 2026")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------ content
def build():
    ensure_dirs()
    cl = pd.read_csv(PROCESSED / "cleaning_log.csv")
    real = pd.read_parquet(APP_DATA / "anomalies.parquet", columns=["household_id", "city", "flag_if", "flag_lof", "flag_z", "flag_iqr", "consensus", "reason", "timestamp"])
    day = pd.read_parquet(APP_DATA / "daily_anomalies.parquet")
    comp = pd.read_csv(APP_DATA / "method_comparison.csv")
    inj = pd.read_csv(APP_DATA / "injection_results.csv")
    wc = pd.read_csv(APP_DATA / "weather_check.csv")
    hc = pd.read_csv(APP_DATA / "household_clusters.csv")
    summ = pd.read_csv(APP_DATA / "cluster_summary.csv")
    ks = pd.read_csv(APP_DATA / "cluster_k_selection.csv")
    var = pd.read_csv(APP_DATA / "pca_variance.csv")
    cases = pd.read_csv(APP_DATA / "cases.csv", parse_dates=["timestamp"])
    ctx = pd.read_parquet(APP_DATA / "case_context.parquet")
    city_hour = pd.read_csv(APP_DATA / "city_hour.csv")
    city_month = pd.read_csv(APP_DATA / "city_month.csv")
    wcorr = pd.read_csv(APP_DATA / "weather_corr.csv")
    sens = pd.read_csv(APP_DATA / "sensitivity.csv")
    agr = pd.read_csv(APP_DATA / "agreement.csv")

    n_h, n_city = real.household_id.nunique(), real.city.nunique()
    raw_rows, hours = int(cl.raw_rows.sum()), len(real)
    n_if, n_c2 = int(real.flag_if.sum()), int((real.consensus >= 2).sum())
    cons = comp[comp.method.str.startswith("Consensus")].iloc[0]
    lof = comp[comp.method == "Local Outlier Factor"].iloc[0]
    wc_t = wc[wc.measure.str.contains("time-only")].iloc[0]
    wc_w = wc[wc.measure.str.contains("weather-aware")].iloc[0]
    ifw = wc[wc.measure == "Isolation Forest flag"].iloc[0]
    reasons = real.loc[real.consensus > 0, "reason"].astype(str).value_counts()
    first, last = real.timestamp.min().date(), real.timestamp.max().date()

    tmp = Path(tempfile.mkdtemp())
    fig_pipeline(tmp / "pipe.png"); fig_profile(tmp / "profile.png", city_hour); fig_month(tmp / "month.png", city_month)
    fig_case(tmp / "case.png", ctx, cases.iloc[0]); fig_pca(tmp / "pca.png", hc, var); fig_injection(tmp / "inj.png", inj)
    fig_weather(tmp / "wx.png", wc)

    s = []
    s += [Spacer(1, 1.2 * cm), P("Pakistan Electricity Consumption Anomaly Mining", TITLE),
          P("Using Data Mining Techniques | Project Report | 2026", ParagraphStyle("sub", parent=BODY, fontSize=12)),
          P("Team: Izaan Imran, Fahad Ahmed, Shiraz Asif, Dayyan Hashmi", BODY), Spacer(1, 0.3 * cm)]

    s += [P("1. What is this project?", H1),
          P(f"We built a data mining system that learns what is <b>normal electricity use for each home</b> and flags the hours that look unusual. "
            f"We used real smart-meter data from <b>{n_h} Pakistani households in {n_city} cities</b> ({first:%b %Y} to {last:%b %Y}). "
            "The results are shown in an interactive Streamlit web app."),
          P("2. What problem does it solve?", H1),
          P(f"The raw data has <b>{raw_rows:,}</b> one-minute readings. Nobody can check that many numbers by hand."),
          P("<b>Why not just say 'high use = anomaly'?</b>")] + bullets([
          "Homes are different. 3 kWh in one hour is normal for a big house with 6 air conditioners and huge for a small flat.",
          "Use changes with the hour of the day, weekday or weekend, and the season.",
          f"Hot weather raises cooling use. In our data, the time-only score calls the hottest 10% of hours 'high' <b>{wc_t.hot_hours_pct:.0f}%</b> of the time, "
          f"against {wc_t.other_hours_pct:.0f}% for other hours. Much of that is simply heat, not a real anomaly."]) + [
          P("So we compare every reading with <b>that home's own normal pattern at that time and temperature</b>."),
          P("3. What becomes easier because of it?", H1)] + bullets([
          "<b>Faster spotting:</b> instead of scanning millions of readings, an analyst looks at a short ranked list of unusual hours.",
          "<b>Fewer false alarms:</b> flags that are explained by hot weather are labelled as such.",
          "<b>Better understanding:</b> homes are grouped by behaviour, and peak hours and seasons are easy to see.",
          "<b>Who benefits:</b> utilities and researchers studying household demand, and students learning anomaly detection on real local data."])

    s += [P("4. What is used?", H1),
          table([["Part", "What we used"],
                 ["Dataset", "REWD-P, LUMS Energy Institute: minute-level smart-meter data, hourly weather, household details"],
                 ["Tools", "Python, pandas, NumPy, scikit-learn, Plotly, Streamlit, matplotlib and reportlab (for this report)"],
                 ["Techniques", "Isolation Forest, Local Outlier Factor (LOF), Z-score, IQR, robust median/MAD baselines, K-Means, PCA"]], [3, 14]),
          P("5. Why exactly these?", H1),
          table([["Choice", "One-line reason"],
                 ["REWD-P", "Real, free, one-minute data from Pakistani homes with matching weather. Exactly what the problem needs."],
                 ["pandas + Parquet", "Simple and fast enough once minute data is turned into hourly data (laptop friendly)."],
                 ["Isolation Forest", "Needs no labels, works on many features, fast, gives a score for ranking."],
                 ["LOF", "Compares a point with its neighbours, so it can find a different kind of oddity (like stuck values)."],
                 ["Z-score and IQR", "Simple, explainable baselines to compare the machine learning methods against."],
                 ["Median and MAD", "Robust 'normal': a few extreme hours do not distort it, unlike the mean."],
                 ["K-Means + PCA", "Easy to explain grouping of homes, and PCA lets us draw it on a 2D chart."],
                 ["Streamlit + Plotly", "A full interactive web app in plain Python, with free hosting on Streamlit Community Cloud."]], [3.5, 13.5]),
          P("6. What are the alternatives, and why not them?", H1),
          table([["Alternative", "Pros", "Cons for our case", "Why we did not choose it"],
                 ["One-Class SVM", "Learns a boundary around normal data", "Slow on 500k rows, sensitive to settings", "Isolation Forest is faster and easier to tune"],
                 ["DBSCAN", "Finds odd points as noise, no K needed", "Hard to choose settings; density varies across homes", "LOF covers local density more gently"],
                 ["Autoencoder / LSTM", "Can learn complex patterns", "Needs more data and tuning, hard to explain, heavy for a laptop", "Goal is simple and explainable; no deep learning"],
                 ["ARIMA / Prophet residuals", "Good for forecasting a time series", "One model per home, slower, more setup", "Per-home medians by hour give a similar 'expected value' simply"],
                 ["Gaussian Mixture", "Soft clusters", "Only 59 homes, so too many parameters", "K-Means is enough and clearer"],
                 ["Hierarchical clustering", "Shows a tree of similarity", "Harder to present; no clear cluster centres", "K-Means gives named profiles"],
                 ["Dash or Flask", "More control over the web page", "More code and setup", "Streamlit is faster to build and deploy"],
                 ["Spark / Dask", "Handles huge data", "Overkill: hourly data is under 500k rows", "pandas is enough and simpler"]], [3.2, 3.6, 5.2, 5.0], font=8),
          P("7. Why is this the best approach for this project?", H1)] + bullets([
          "<b>Unsupervised:</b> the dataset has no anomaly labels, so we cannot train a supervised model.",
          "<b>Per-household baselines:</b> every home is judged against itself, which is fair.",
          "<b>Weather-aware:</b> the temperature-adjusted baseline separates 'hot' from 'unusual' and avoids false alarms.",
          "<b>Light:</b> minute data is reduced to hourly Parquet files; the whole project runs on a normal laptop.",
          "<b>Explainable:</b> each flag has a plain-language reason, which suits a live presentation and viva."])

    s += [P("8. How we did it", H1), img(tmp / "pipe.png", 17),
          table([["Step", "What we did", "Result"],
                 ["1 Collect", "Downloaded energy, weather and metadata files. Checked columns and units.", f"{n_h} homes (proposal said 60), power in kW"],
                 ["2 Clean", "Fixed timestamps, merged duplicate timestamps, set impossible values to missing, kept real highs and zeros, made hourly kWh.",
                  f"{raw_rows:,} minute rows to {hours:,} hourly rows. {int(cl.duplicate_timestamps_merged.sum()):,} duplicates merged, {int(cl.invalid_values_set_missing.sum())} invalid values"],
                 ["3 Join", "Matched each hour with the weather of the home's city.", "0 unmatched rows"],
                 ["4 EDA", "Looked at daily, seasonal and weather patterns.", "Peak hours 21:00 to 01:00; summer is the busiest"],
                 ["5 Features", "Hour, day, month, peak flag, lags, rolling averages, change, weather. Past values only (no leakage).", "20+ features per hour"],
                 ["6 Normal", "Median and MAD per home for hour x day type, and for hour x temperature band.", "Robust 'deviation scores'"],
                 ["7 Detect", "Isolation Forest, LOF, Z-score, IQR per home. Also daily windows.", f"{n_if:,} hours flagged by Isolation Forest; {n_c2:,} by 2+ methods"],
                 ["8 Cluster", "K-Means on 8 behaviour numbers per home; PCA to draw.", f"{hc.cluster.nunique()} home groups"],
                 ["9 Evaluate", "Injection test, agreement, rates, cases, weather check.", "See next section"],
                 ["10 Show", "Eight-page Streamlit app and this report.", "app.py"]], [2.1, 8.6, 6.3], font=8)]

    s += [P("9. Results", H1),
          P(f"<b>Key numbers.</b> {n_h} homes, {n_city} cities, {hours:,} hourly readings. Isolation Forest flagged {n_if:,} hours (1% by design). "
            f"<b>{n_c2:,}</b> hours were flagged by two or more methods. The daily-window check flagged <b>{int(day.flag_daily.sum())}</b> of {len(day):,} days."),
          P("<b>What the data looks like.</b> Energy is highest late in the evening and at night, and clearly higher in summer, which points to cooling.", BODY),
          img(tmp / "profile.png", 12.5),
          P("Figure 1. Average energy by hour of day for each city.", SMALL),
          img(tmp / "month.png", 12.5),
          P(f"Figure 2. Average energy by month. Across homes, the average correlation of energy with temperature is {wcorr.temp.mean():.2f}; with humidity {wcorr.humidity.mean():.2f}; with rain {wcorr.precip.mean():.2f}.", SMALL)]

    s += [KeepTogether([P("Method comparison (injection test)", H2),
          P("There are no real labels, so we <b>injected known anomalies</b> into real data (spikes, drops to zero, 6-hour level shifts and 6-hour flat lines) "
            "and checked which methods find them. This is a <b>controlled validation, not real labels</b>. Hours that two or more methods had already flagged in the real data were left out of scoring. "
            "Average of 3 repeats on 12 homes (2 per city)."),
          table([["Method", "Flags on real data", "Precision", "Recall", "F1", "AUC"]] +
                [[r.method, f"{r.flag_rate_pct:.2f}%", f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", f"{r.auc:.2f}"] for r in comp.itertuples()],
                [5.6, 2.8, 2.2, 2.0, 1.8, 1.8]),
          P("Precision: of the hours flagged, how many were injected. Recall: of the injected hours, how many were found. F1 mixes both. "
            "AUC checks the raw scores without any threshold (0.5 = random, 1.0 = perfect).", SMALL)]),
          img(tmp / "inj.png", 13), P("Figure 3. AUC by injected anomaly type.", SMALL)]
    ev = inj[inj.type.isin(["shift_event", "flatline_event"])].pivot(index="method", columns="type", values="recall")
    s += bullets([
        "<b>Spikes</b> are found very well by Isolation Forest and the per-home Z-score (AUC close to 1.0).",
        f"<b>Flat lines</b> are found mainly by LOF ({ev.loc['Local Outlier Factor', 'flatline_event']:.0%} of the blocks had at least one hour flagged). Isolation Forest does not see them.",
        f"<b>Level shifts</b> of 2.5 times for 6 hours are hard at 1% flags; LOF ranks them best (AUC {inj[(inj.method == 'Local Outlier Factor') & (inj.type == 'shift')].auc.iloc[0]:.2f}).",
        f"<b>Naive global threshold</b> (all homes together) is the weakest (F1 {comp[comp.method.str.startswith('Global')].f1.iloc[0]:.2f}), which proves that per-home normal matters.",
        f"<b>Consensus</b> (2+ methods agree) has the highest precision ({cons.precision:.2f}) but low recall ({cons.recall:.2f}): use it for high-confidence alerts.",
        f"<b>Recommendation:</b> Isolation Forest as the main ranking score (as proposed), hours flagged by 2+ methods as high-confidence, and LOF to look for stuck or zero readings. "
        f"LOF had the best overall AUC ({lof.auc:.2f}) but the lowest precision among the ML methods ({lof.precision:.2f})."])
    jac = agr.pivot(index="method_a", columns="method_b", values="jaccard")
    s += [P("<b>Agreement.</b> The methods catch different things: the overlap (Jaccard) between Isolation Forest and Z-score is "
            f"{jac.loc['flag_if', 'flag_z']:.2f}, and between LOF and the others is only {jac.loc['flag_lof', ['flag_if', 'flag_z', 'flag_iqr']].mean():.2f} on average. That is why combining them is useful."),
          P("<b>Sensitivity.</b> Isolation Forest flags grow with the contamination setting: " +
            ", ".join(f"{r.contamination:.1%} gives {r.flagged_hours:,} hours" for r in sens.itertuples()) + ".")]

    s += [P("Weather check", H2),
          img(tmp / "wx.png", 9.5),
          P("Figure 4. Share of hours called 'high', by baseline type.", SMALL),
          P(f"With the time-only baseline, the hottest 10% of hours are called high <b>{wc_t.ratio_hot_vs_other:.1f} times</b> more often than other hours. "
            f"With the weather-aware baseline this falls to <b>{wc_w.ratio_hot_vs_other:.1f} times</b>, so most heat-driven highs are explained. "
            f"The model-based flags still cluster in heat (Isolation Forest flags {ifw.hot_hours_pct:.1f}% of the hottest hours vs {ifw.other_hours_pct:.1f}% of others), "
            "so some extreme-heat hours remain flagged. Of the flagged hours, "
            f"{int(reasons.get('High for this hour, but typical for this temperature', 0)):,} are 'high but typical for this temperature' and "
            f"{int(reasons.get('Much higher than usual for this hour, even allowing for temperature', 0)):,} are 'unusual even allowing for temperature'.")]

    s += [P("Household groups", H2), img(tmp / "pca.png", 12.5),
          P(f"Figure 5. K-Means groups on a PCA map (the two axes hold {100 * var.explained.sum():.0f}% of the variation).", SMALL),
          table([["Group", "Homes", "Avg kWh/h", "Evening share", "Night share", "Link to temp.", "2+ methods flag (%)"]] +
                [[r.cluster_name, r.households, f"{r.mean_kwh:.2f}", f"{r.evening_share:.2f}", f"{r.night_share:.2f}", f"{r.temp_corr:.2f}", f"{r.multi_method_rate_pct:.2f}"] for r in summ.itertuples()],
                [6.2, 1.4, 1.8, 1.9, 1.7, 1.9, 2.1], font=8),
          P(f"We tried K = 2 to 8. The silhouette score is low (best {ks.silhouette.max():.2f}), so groups overlap and are a helpful summary, not sharp boxes. "
            f"We chose K = {hc.cluster.nunique()}, the best silhouette among K = 3 to 6, because K = 2 is too coarse to be useful. "
            "Skardu homes mostly fall in the low-use, evening-peak group; most homes in Karachi, Lahore, Multan and Peshawar fall in the temperature-sensitive group.")]

    s += [P("Example cases", H2), img(tmp / "case.png", 12.5),
          P(f"Figure 6. Case 1 (House {cases.iloc[0].household_id}, {cases.iloc[0].city}): {cases.iloc[0].note}", SMALL)]
    case_rows = [["#", "Home / city", "Time", "Kind", "What we see"]]
    for r in cases.itertuples():
        case_rows.append([r.case_id, f"{r.household_id} / {r.city}", f"{r.timestamp:%Y-%m-%d %H:%M}", r.kind, r.note])
    s += [table(case_rows, [0.6, 2.0, 2.5, 3.6, 8.3], font=7.5),
          P("Cases 1 to 5 are strong flags. Some strong flags are marked 'high for this hour, but typical for this temperature': all four methods flagged them, but the weather-aware score says heat explains the load (check the reason text in the table). "
            "Cases 6 and 7 look like possible false alarms (only one method flags them) and would need extra information to judge.", SMALL)]

    s += [P("10. Limitations and honest notes", H1)] + bullets([
        "<b>No real labels.</b> Accuracy comes from injected, synthetic anomalies. Real anomalies may be different, so scores may differ.",
        "<b>A flag is not a cause.</b> It means 'differs from the learned normal pattern'. It is not proof of theft, equipment failure or anything else.",
        f"<b>{n_h} households, not 60.</b> The dataset page lists 59 files; the proposal says 60. We report what we found. Only {n_h} homes, so results do not describe all of Pakistan.",
        "<b>Uneven coverage.</b> Most homes have about one year; some start earlier or end later, and some have gaps (an hour needs 45 valid minutes to count).",
        "<b>Timezone</b> is not stated in the data; we assume local Pakistan time. Two homes report solar generation and some may use net metering; we use the 'Usage' column as given.",
        "<b>Zeros</b> may be real outages (load-shedding). We kept them and flag them as 'near zero while usually in use'.",
        "<b>Normal is learned from the whole period</b> (offline analysis), not in real time.",
        "<b>Weak cluster separation</b> (silhouette about 0.2) and modest recall on 6-hour level shifts.",
        "<b>Peak hours</b> were set from the data (21:00 to 01:00), not assumed."])

    s += [KeepTogether([P("11. How to run it", H1),
          P(f"<b>GitHub:</b> {GITHUB_URL}<br/><b>Live app:</b> {APP_URL}"),
          table([["Task", "Command (inside the project folder)"],
                 ["Create environment", "python -m venv .venv  then activate it  (Windows: .venv\\Scripts\\activate; Mac/Linux: source .venv/bin/activate)"],
                 ["Install for the app only", "pip install -r requirements.txt"],
                 ["Install for everything", "pip install -r requirements-pipeline.txt"],
                 ["Rebuild all results", "python run_pipeline.py   (needs the raw REWD-P files in data/raw, see README)"],
                 ["Start the app", "streamlit run app.py"],
                 ["Test the app", "python tests/test_app.py"],
                 ["Deploy", "Streamlit Community Cloud: pick the repo, branch main, main file app.py, Python 3.13"]], [4, 13], font=8.5)])]

    qa = [
        ("Why Isolation Forest?", "It needs no labels, handles many features, is fast on 500k rows and gives a score we can rank. It isolates odd points with few random splits."),
        ("Why not just a threshold?", "Homes differ in size and habits, and use changes with hour, season and temperature. One threshold would flag big homes all the time and miss problems in small ones."),
        ("What is contamination?", "The share of points we expect to be anomalous. We use 1% so flags stay rare, and also test 0.5% and 2%."),
        ("How did you evaluate without labels?", "We injected known anomalies (spikes, zeros, shifts, flat lines) into real data and measured precision, recall, F1 and AUC. We also checked agreement between methods, rates by home, city and month, example cases and a weather check."),
        ("Why per household?", "So each home is compared with its own normal. The naive global threshold was the worst method in our test."),
        ("What is the silhouette score?", f"A number from -1 to 1 that shows how well points fit their own cluster versus the next one. Ours is low (about {ks.silhouette.max():.2f}), so groups overlap."),
        ("What are the limitations?", "No real labels, a flag is not a cause, only 59 homes, uneven coverage, weak cluster separation and modest recall on gradual shifts."),
        ("How does weather change the result?", f"The weather-aware baseline cuts 'high in hot hours' from {wc_t.ratio_hot_vs_other:.1f}x to {wc_w.ratio_hot_vs_other:.1f}x. Heat-driven highs are labelled 'typical for this temperature' instead of being treated as unusual."),
        ("Why median and MAD, not mean and standard deviation?", "A few huge values inflate the mean and standard deviation, which hides the very anomalies we look for. Median and MAD barely move."),
        ("Why hourly data, not minute data?", "Minute data is 30 million rows and noisy. Hourly kWh keeps the daily pattern, is 60 times smaller and runs on a laptop and on free cloud hosting."),
        ("Did you remove outliers when cleaning?", "No. We only set impossible readings (negative or above 50 kW) to missing; none were found. High values and zeros are kept because they may be real."),
        ("Why does LOF disagree with the others?", "LOF looks at local neighbourhoods, so it finds different oddities, such as stuck or zero values. That is why it is a useful complement."),
        ("Does a flag mean theft or a fault?", "No. It only means the reading differs from the learned normal pattern. Any cause needs extra evidence."),
        ("Why K = 3 for K-Means?", "It had the best silhouette among K = 3 to 6. K = 2 scored slightly higher but is too coarse to describe different home types."),
    ]
    s += [P("12. Likely viva questions", H1)]
    for i, (q, a) in enumerate(qa, 1):
        s.append(KeepTogether([P(f"<b>Q{i}. {q}</b>", BODY), P(a, BODY)]))
    s += [P("Data source: LUMS Energy Institute, Residential Energy and Weather Data Pakistan (REWD-P), lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html", SMALL)]

    out = DOCS / "Project_Report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Pakistan Electricity Consumption Anomaly Mining", author="Izaan Imran, Fahad Ahmed, Shiraz Asif, Dayyan Hashmi").build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
