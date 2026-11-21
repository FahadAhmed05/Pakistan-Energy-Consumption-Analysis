"""Pakistan Electricity Consumption Anomaly Mining - Streamlit app.

The app only READS small precomputed files from app_data/ (made by run_pipeline.py).
It never trains a model while you click around, so every page is fast.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DATA = Path(__file__).parent / "app_data"
NORMAL, ANOMALY, GREY = "#1f6f8b", "#e8710a", "#9aa7b0"
CHART_H = 380
DISCLAIMER = ("An anomaly flag only means **\"this reading differs from the learned normal pattern\"**. "
              "It is **not proof** of electricity theft, equipment failure or any other specific cause. "
              "That would need extra evidence and a proper investigation.")
METHODS = {"Isolation Forest (main method)": "flag_if", "Local Outlier Factor": "flag_lof",
           "Z-score (per home & hour)": "flag_z", "IQR (per home & hour)": "flag_iqr",
           "Consensus (2 or more methods agree)": "consensus2"}
SCORES = {"flag_if": "if_score", "flag_lof": "lof_score"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Pakistan Electricity Anomaly Mining", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------- data loading (cached)
@st.cache_data(show_spinner="Loading data ...")
def load_hourly():
    df = pd.read_parquet(APP_DATA / "anomalies.parquet")
    df["city"] = df["city"].astype(str)
    df["reason"] = df["reason"].astype(str)
    df["consensus2"] = df["consensus"] >= 2
    return df


@st.cache_data
def load_csv(name):
    return pd.read_csv(APP_DATA / name)


@st.cache_data
def load_parquet(name):
    return pd.read_parquet(APP_DATA / name)


def house_label(hid, city):
    return f"House {hid} ({city})"


def house_picker(key, default=None):
    """Sidebar-free household dropdown. Returns the household id."""
    s = load_csv("household_summary.csv")
    labels = {house_label(r.household_id, r.city): int(r.household_id) for r in s.itertuples()}
    names = list(labels)
    idx = 0 if default is None else names.index(house_label(default, s.loc[s.household_id == default, "city"].iloc[0]))
    return labels[st.selectbox("Household", names, index=idx, key=key)]


def how_to_read(text):
    st.caption(f"**How to read this chart:** {text}")


def style(fig, ytitle=None, xtitle=None, height=CHART_H):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12),
                      yaxis_title=ytitle, xaxis_title=xtitle, template="plotly_white")
    return fig


def empty_msg(msg="No data for this selection. Try another household or a wider date range."):
    st.info(msg)


# ---------------------------------------------------------------- pages
def page_home():
    df = load_hourly()
    st.title("⚡ Pakistan Electricity Consumption Anomaly Mining")
    st.markdown("**Team:** Izaan Imran · Fahad Ahmed · Shiraz Asif · Dayyan Hashmi  |  **Data Mining project, 2026**")
    st.markdown(
        "Homes use electricity in different ways, and a single reading can look high or low for many normal reasons "
        "(hot weather, a guest, an appliance). We learn what is **normal for each home** and flag the hours that differ "
        "from it, using real one-minute smart-meter data from 59 Pakistani households.")
    c = st.columns(5)
    c[0].metric("Households", df.household_id.nunique())
    c[1].metric("Cities", df.city.nunique())
    c[2].metric("Hours analysed", f"{len(df):,}")
    c[3].metric("Flagged by Isolation Forest", f"{int(df.flag_if.sum()):,}")
    c[4].metric("Flagged by 2+ methods", f"{int(df.consensus2.sum()):,}")
    comp = load_csv("method_comparison.csv").set_index("method")
    best = comp["auc"].idxmax()
    st.success(f"**Best method in our controlled test:** {best} (AUC {comp.loc[best, 'auc']:.2f}). "
               "No single method wins everywhere, so the Method Comparison page explains which to use for what.")
    st.markdown("#### How to use this app")
    st.markdown("Pick a page on the left. Start with **Anomaly Detector** to see flagged hours for any home, "
                "then **Weather Impact** to see which flags are explained by temperature.")
    st.warning(DISCLAIMER)


def page_explorer():
    st.header("Data Explorer")
    df = load_hourly()
    c1, c2, c3 = st.columns([1, 1, 2])
    city = c1.selectbox("City", ["All cities"] + sorted(df.city.unique()))
    d = df if city == "All cities" else df[df.city == city]
    ids = sorted(d.household_id.unique())
    hid = c2.selectbox("Household", ["All in selection"] + [int(i) for i in ids])
    if hid != "All in selection":
        d = d[d.household_id == hid]
    lo, hi = d.timestamp.min().date(), d.timestamp.max().date()
    rng = c3.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if len(rng) != 2:
        empty_msg("Pick both a start and an end date."); return
    d = d[(d.timestamp.dt.date >= rng[0]) & (d.timestamp.dt.date <= rng[1])]
    if d.empty:
        empty_msg(); return
    t1, t2, t3, t4 = st.tabs(["Daily pattern", "Over time", "Seasons", "Weather link"])
    with t1:
        prof = d.groupby(["hour", "is_weekend"])["kwh"].mean().reset_index()
        prof["Day type"] = prof["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        fig = px.line(prof, x="hour", y="kwh", color="Day type", color_discrete_sequence=[NORMAL, GREY], markers=True)
        st.plotly_chart(style(fig, "Average energy (kWh per hour)", "Hour of day"), width="stretch")
        how_to_read("each point is the average energy used in that hour of the day. Higher points are the busy hours (peaks).")
    with t2:
        day = d.assign(date=d.timestamp.dt.floor("D")).groupby("date")["kwh"].mean().reset_index()
        fig = px.line(day, x="date", y="kwh", color_discrete_sequence=[NORMAL])
        st.plotly_chart(style(fig, "Average energy (kWh per hour)", "Date"), width="stretch")
        how_to_read("one point per day (average hourly use that day). Bumps in summer usually mean cooling.")
    with t3:
        mon = d.groupby("month")["kwh"].mean().reindex(range(1, 13)).reset_index()
        mon["Month"] = MONTHS
        fig = px.bar(mon, x="Month", y="kwh", color_discrete_sequence=[NORMAL])
        st.plotly_chart(style(fig, "Average energy (kWh per hour)"), width="stretch")
        how_to_read("average use per calendar month. Months with no bar have no data for this selection.")
    with t4:
        corr = load_csv("weather_corr.csv")
        corr = corr if city == "All cities" else corr[corr.city == city]
        s = d.sample(min(len(d), 6000), random_state=42)
        fig = px.scatter(s, x="temp_c", y="kwh", opacity=0.35, color_discrete_sequence=[NORMAL])
        st.plotly_chart(style(fig, "Energy (kWh per hour)", "Outdoor temperature (°C)"), width="stretch")
        how_to_read("each dot is one hour. If dots rise to the right, the home uses more energy when it is hot.")
        st.write(f"Average correlation of energy with temperature for this selection: **{corr['temp'].mean():.2f}** "
                 f"(humidity {corr['humidity'].mean():.2f}, rain {corr['precip'].mean():.2f}). Closer to 1 means a stronger link.")


def page_normal():
    st.header("Normal Behaviour")
    st.write("This is what a typical day looks like for a home. The shaded band holds the middle 80% of its days at that hour.")
    hid = house_picker("normal_house")
    prof = load_csv("profile_hourly.csv")
    p = prof[prof.household_id == hid]
    if p.empty:
        empty_msg(); return
    cols = st.columns(2)
    for col, wk, title in [(cols[0], 0, "Weekdays"), (cols[1], 1, "Weekends")]:
        q = p[p.is_weekend == wk].sort_values("hour")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q.hour, y=q.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.hour, y=q.p10, fill="tonexty", fillcolor="rgba(31,111,139,0.18)", line=dict(width=0), name="Normal band (10%-90%)"))
        fig.add_trace(go.Scatter(x=q.hour, y=q.normal, line=dict(color=NORMAL, width=3), name="Typical (median)"))
        fig.update_layout(title=title)
        col.plotly_chart(style(fig, "Energy (kWh per hour)", "Hour of day"), width="stretch")
    how_to_read("the dark line is the typical use at each hour. A reading far outside the shaded band is unusual for this home.")


def page_detector():
    st.header("Anomaly Detector")
    st.warning(DISCLAIMER)
    df = load_hourly()
    c1, c2 = st.columns(2)
    with c1:
        hid = house_picker("det_house")
    method_name = c2.selectbox("Method", list(METHODS))
    flag = METHODS[method_name]
    h = df[df.household_id == hid].sort_values("timestamp")
    if h.empty:
        empty_msg(); return
    top = h.loc[h["if_score"].idxmax(), "timestamp"]
    lo, hi = h.timestamp.min().date(), h.timestamp.max().date()
    start = max(lo, (top - pd.Timedelta(days=15)).date())
    end = min(hi, (top + pd.Timedelta(days=15)).date())
    rng = st.date_input("Date range (starts around this home's strongest anomaly)", (start, end), min_value=lo, max_value=hi)
    if len(rng) != 2:
        empty_msg("Pick both a start and an end date."); return
    w = h[(h.timestamp.dt.date >= rng[0]) & (h.timestamp.dt.date <= rng[1])]
    if w.empty:
        empty_msg(); return
    a = w[w[flag]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Hours shown", f"{len(w):,}")
    m2.metric("Flagged hours", f"{len(a):,}")
    m3.metric("Share flagged", f"{100 * len(a) / len(w):.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.kwh, mode="lines", line=dict(color=NORMAL, width=1.4), name="Energy used"))
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.base_med, mode="lines", line=dict(color=GREY, width=1, dash="dot"), name="Normal for this hour"))
    fig.add_trace(go.Scatter(x=a.timestamp, y=a.kwh, mode="markers", marker=dict(color=ANOMALY, size=9, symbol="diamond"), name="Flagged"))
    st.plotly_chart(style(fig, "Energy (kWh per hour)", None, 420), width="stretch")
    how_to_read("the blue line is real use. The grey dotted line is what is normal for that hour. Orange diamonds are flagged hours.")

    if flag in SCORES:
        fig2 = px.line(w, x="timestamp", y=SCORES[flag], color_discrete_sequence=[GREY])
        fig2.add_trace(go.Scatter(x=a.timestamp, y=a[SCORES[flag]], mode="markers", marker=dict(color=ANOMALY, size=7), name="Flagged"))
        st.plotly_chart(style(fig2, "Anomaly score (higher = more unusual)", None, 260), width="stretch")
        how_to_read("the model's score for each hour. Flagged hours are the ones with the highest scores.")
    t1, t2 = st.tabs(["Flagged hours", "Whole days that look unusual"])
    with t1:
        if a.empty:
            st.info("No hours were flagged in this window. Try a wider date range or another method.")
        else:
            tab = a[["timestamp", "kwh", "base_med", "temp_c", "reason", "consensus"]].rename(columns={
                "timestamp": "Time", "kwh": "Used (kWh)", "base_med": "Normal (kWh)", "temp_c": "Temp (°C)",
                "reason": "Why it was flagged", "consensus": "Methods agreeing (of 4)"})
            st.dataframe(tab.round(2), width="stretch", hide_index=True)
            st.download_button("Download flagged hours (CSV)", tab.to_csv(index=False).encode(), f"house_{hid}_flagged.csv", "text/csv")
    with t2:
        day = load_parquet("daily_anomalies.parquet")
        day = day[(day.household_id == hid) & day.flag_daily]
        if day.empty:
            st.info("No unusual days for this household.")
        else:
            st.dataframe(day[["date", "total_kwh", "peak_kwh", "daily_reason"]].rename(columns={
                "date": "Day", "total_kwh": "Total (kWh/day)", "peak_kwh": "Highest hour (kWh)", "daily_reason": "Why"}).round(2),
                width="stretch", hide_index=True)
        how_to_read("a day is flagged when its total or shape is unusual for this home (checked with Isolation Forest on daily features).")


def page_clusters():
    st.header("Household Clusters")
    st.write("Households are grouped by how they use electricity (size, time of day, weekend pattern, sensitivity to heat).")
    hc = load_csv("household_clusters.csv")
    summ = load_csv("cluster_summary.csv")
    var = load_csv("pca_variance.csv")
    fig = px.scatter(hc, x="pc1", y="pc2", color="cluster_name", hover_data=["household_id", "city", "mean_kwh"],
                     color_discrete_sequence=px.colors.qualitative.Safe, symbol="city")
    st.plotly_chart(style(fig, f"PC2 ({100 * var.explained[1]:.0f}% of variation)", f"PC1 ({100 * var.explained[0]:.0f}% of variation)", 460), width="stretch")
    how_to_read("each dot is a household. Close dots behave alike. PCA squeezes 8 behaviour numbers into 2 axes just for drawing.")
    t1, t2, t3 = st.tabs(["Group profiles", "Cities in each group", "How K was chosen"])
    with t1:
        st.dataframe(summ.rename(columns={"cluster_name": "Group", "households": "Homes", "mean_kwh": "Average kWh/h",
                     "peak_to_avg": "Peak ÷ average", "evening_share": "Evening share", "night_share": "Night share",
                     "temp_corr": "Link to temperature", "multi_method_rate_pct": "Hours flagged by 2+ methods (%)"}),
                     width="stretch", hide_index=True)
        st.caption("Isolation Forest flags about 1% of hours in every home by design, so we compare groups by how often two or more methods agree.")
    with t2:
        st.dataframe(pd.crosstab(hc.cluster_name, hc.city), width="stretch")
    with t3:
        k = load_csv("cluster_k_selection.csv")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k.k, y=k.silhouette, name="Silhouette (higher is better)", line=dict(color=NORMAL)))
        fig.add_trace(go.Scatter(x=k.k, y=k.inertia, name="Inertia (elbow)", yaxis="y2", line=dict(color=ANOMALY, dash="dot")))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(style(fig, "Silhouette", "Number of groups K", 320), width="stretch")
        st.caption("Silhouette is low (about 0.2), so groups overlap: they are a helpful summary, not sharp boxes. We chose K = 3, "
                   "the best silhouette among K = 3 to 6, because K = 2 is too coarse to be useful.")


def page_weather():
    st.header("Weather Impact")
    df = load_hourly()
    st.write("Hot weather raises cooling use, so a high reading is not always unusual. "
             "Here we separate **flags explained by temperature** from **flags that are unusual even allowing for temperature**.")
    t1, t2, t3 = st.tabs(["Energy vs temperature", "Explained vs truly unusual", "Heat check"])
    with t1:
        hid = house_picker("wx_house")
        h = df[df.household_id == hid]
        s = h.sample(min(len(h), 6000), random_state=42)
        s = s.assign(Status=s.flag_if.map({True: "Flagged (Isolation Forest)", False: "Normal"}))
        fig = px.scatter(s, x="temp_c", y="kwh", color="Status", opacity=0.5,
                         color_discrete_map={"Normal": NORMAL, "Flagged (Isolation Forest)": ANOMALY})
        st.plotly_chart(style(fig, "Energy (kWh per hour)", "Outdoor temperature (°C)"), width="stretch")
        how_to_read("each dot is an hour. Flagged hours (orange) that sit where the cloud rises with heat are less surprising.")
        tr = load_csv("temp_response.csv")
        tr = tr[tr.household_id == hid]
        if not tr.empty:
            st.plotly_chart(style(px.line(tr, x="temp_bin", y="mean_kwh", markers=True, color_discrete_sequence=[NORMAL]),
                                  "Average kWh per hour", "Temperature band (°C, 2-degree steps)", 280), width="stretch")
    with t2:
        f = df[df.consensus > 0]
        rc = f.reason.value_counts().reset_index()
        rc.columns = ["Reason", "Hours"]
        fig = px.bar(rc, x="Hours", y="Reason", orientation="h", color_discrete_sequence=[NORMAL])
        st.plotly_chart(style(fig, None, "Flagged hours (any method)", 340), width="stretch")
        n_exp = int((f.reason == "High for this hour, but typical for this temperature").sum())
        n_unu = int((f.reason == "Much higher than usual for this hour, even allowing for temperature").sum())
        st.write(f"Of the flagged hours that were high, **{n_exp:,}** are typical for the temperature and **{n_unu:,}** stay unusual even after allowing for it.")
        how_to_read("every flagged hour gets one plain-language reason. 'Typical for this temperature' means the weather-aware baseline does not find it unusual.")
    with t3:
        wc = load_csv("weather_check.csv")
        st.dataframe(wc.rename(columns={"measure": "Measure", "hot_hours_pct": "In the hottest 10% of hours (%)",
                     "other_hours_pct": "All other hours (%)", "ratio_hot_vs_other": "Ratio"}), width="stretch", hide_index=True)
        st.caption("The time-only score calls hot hours 'high' 2.1 times more often. The weather-aware score cuts that to 1.1 times, "
                   "so most heat-driven highs are explained. The model-based flags still cluster in heat (about 3 times), so some "
                   "extreme-heat hours remain flagged.")


def page_methods():
    st.header("Method Comparison & Evaluation")
    st.write("There are no true anomaly labels in this dataset, so we tested the methods by **injecting known anomalies into real data** "
             "(a controlled test, not real labels) and checking which methods find them.")
    comp = load_csv("method_comparison.csv")
    st.subheader("Summary table")
    st.dataframe(comp.rename(columns={"method": "Method", "flag_rate_pct": "Flags on real data (%)", "precision": "Precision",
                 "recall": "Recall", "f1": "F1", "auc": "AUC (ranking quality)"}), width="stretch", hide_index=True)
    st.caption("Precision: of the hours flagged, how many were truly injected. Recall: of the injected hours, how many were found. "
               "F1 combines both. AUC measures how well the raw score ranks injected hours above normal ones, whatever threshold is used.")
    inj = load_csv("injection_results.csv")
    t1, t2, t3, t4 = st.tabs(["By anomaly type", "Agreement", "Sensitivity", "Which method?"])
    with t1:
        metric = st.radio("Metric", ["auc", "recall", "f1"], horizontal=True, format_func=lambda m: {"auc": "AUC", "recall": "Recall", "f1": "F1"}[m])
        d = inj[inj.type.isin(["spike", "drop", "shift", "flatline"])]
        fig = px.bar(d, x="type", y=metric, color="method", barmode="group", color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(style(fig, metric.upper(), "Injected anomaly type", 420), width="stretch")
        how_to_read("spike = one very high hour, drop = sudden zero, shift = 6 hours at 2.5x, flatline = 6 hours stuck at one value. Taller is better.")
    with t2:
        agr = load_csv("agreement.csv")
        pv = agr.pivot(index="method_a", columns="method_b", values="jaccard")
        names = {"flag_if": "Isolation Forest", "flag_lof": "LOF", "flag_z": "Z-score", "flag_iqr": "IQR"}
        pv = pv.rename(index=names, columns=names)
        fig = px.imshow(pv, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1)
        st.plotly_chart(style(fig, None, None, 380), width="stretch")
        how_to_read("1.0 means two methods flag exactly the same hours, 0 means no overlap. Low numbers show the methods catch different things, which is why combining them helps.")
    with t3:
        st.dataframe(load_csv("sensitivity.csv").rename(columns={"contamination": "Contamination (expected share)",
                     "flagged_hours": "Flagged hours", "share_pct": "Share (%)"}), width="stretch", hide_index=True)
        st.caption("Contamination is the share of hours we expect to be anomalous. Flags grow in step with it, so choose it to match how many alerts you can inspect.")
    with t4:
        st.markdown(
            "- **Isolation Forest (main method):** very good at spikes, gives a score for ranking, fast. Weaker on frozen-meter patterns.\n"
            "- **Local Outlier Factor:** best overall ranking in the test and the only one that notices flat-lined or zero hours, "
            "but it often disagrees with the others and has the lowest precision.\n"
            "- **Z-score / IQR per home and hour:** simple and easy to explain. Z-score is precise; IQR flags too many hours (5%).\n"
            "- **Global z-score:** ignores that homes differ and performs worst. This is why we learn normal per household.\n"
            "- **Recommendation:** use Isolation Forest to rank unusual hours, treat hours flagged by **2 or more methods** as high confidence "
            "(precision above 0.9 in the test), and use LOF to look for stuck or zero readings.")


def page_about():
    st.header("About & Limitations")
    st.markdown(
        "**Data:** Residential Energy and Weather Data Pakistan (REWD-P), LUMS Energy Institute: "
        "[lei.lums.edu.pk](https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html). "
        "Smart-meter readings every minute from households in Lahore, Multan, Peshawar, Karachi, Islamabad and Skardu, plus hourly weather.")
    st.markdown("#### What we did, in simple words")
    st.markdown(
        "1. Cleaned the minute data and turned it into **hourly energy** (kWh).\n"
        "2. Added the **weather** of the home's city for each hour.\n"
        "3. Learned each home's **normal pattern** (typical use for each hour, weekday or weekend, and each temperature range).\n"
        "4. Measured how far every hour is from normal and flagged unusual hours with four methods.\n"
        "5. Grouped similar homes with K-Means and tested the methods by injecting known anomalies.")
    st.markdown("#### Limitations")
    st.markdown(
        "- The data has **no real anomaly labels**, so accuracy is estimated with injected, synthetic anomalies. Scores on real anomalies may differ.\n"
        "- We found **59 households, not 60** as the proposal says (the dataset page lists 59). Coverage differs by home: most have about one year, a few have gaps.\n"
        "- Timestamps have no timezone in the source. We assume local Pakistan time.\n"
        "- Two homes report solar generation and some may use net metering. We use the 'Usage' column as given.\n"
        "- Zero readings may be real outages (load-shedding). We keep them and flag them as 'near zero'.\n"
        "- Each home's normal pattern is learned from the whole recorded period (offline analysis, not live monitoring).\n"
        "- Cluster separation is modest (silhouette about 0.2).\n"
        "- Only 59 homes, so results describe this sample and not all of Pakistan.")
    st.warning(DISCLAIMER)


PAGES = {"Home": page_home, "Data Explorer": page_explorer, "Normal Behaviour": page_normal,
         "Anomaly Detector": page_detector, "Household Clusters": page_clusters, "Weather Impact": page_weather,
         "Method Comparison": page_methods, "About & Limitations": page_about}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.caption("Colours: blue = normal, orange = flagged.")
PAGES[choice]()
