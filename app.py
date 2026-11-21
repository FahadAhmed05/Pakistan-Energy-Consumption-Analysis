"""Pakistan Electricity Consumption Anomaly Mining - Streamlit app.

The app only READS small precomputed files from app_data/ (made by run_pipeline.py).
It never trains a model while you click around, so every page is fast.

Design: one validated colour pair (blue = normal, orange = flagged), insight titles
instead of generic chart titles, a table view behind every chart, and a Back / Next
footer so the pages can be presented in story order.
"""
import itertools
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
APP_DATA = ROOT / "app_data"

# ---- design tokens ----
# Data colours are mid-tones validated on BOTH the light (#fcfcfb) and dark (#1a1a19) surface.
# Text, grid lines and marker rings are recoloured per theme by assets/style.css (CSS light-dark()).
NORMAL, ANOMALY = "#3482e0", "#e2602e"                 # blue = normal, orange = flagged
INK, INK2, MUTED = "#0b0b0b", "#7a7a75", "#898781"
GRID, BASELINE, SURFACE = "rgba(128,128,128,0.25)", "rgba(128,128,128,0.45)", "rgba(128,128,128,0.5)"
CLUSTER_COLORS = ["#3482e0", "#d8628c", "#0b8a0b"]       # blue, magenta, green (orange is reserved for flags)
CLUSTER_SYMBOLS = ["circle", "diamond", "square"]
SEQUENTIAL = [[0, "#cde2fb"], [0.35, "#86b6ef"], [0.65, "#3987e5"], [0.85, "#1c5cab"], [1, "#0d366b"]]
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
PLOT_CONFIG = {"displayModeBar": False}

DISCLAIMER = ("An anomaly flag only means <b>“this reading differs from the learned normal pattern”</b>. "
              "It is <b>not proof</b> of electricity theft, equipment failure or any other specific cause. "
              "That would need extra evidence and a proper investigation.")
METHODS = {"Isolation Forest (main method)": "flag_if", "Local Outlier Factor": "flag_lof",
           "Z-score (per home & hour)": "flag_z", "IQR (per home & hour)": "flag_iqr",
           "Consensus (2 or more methods agree)": "consensus2"}
SCORES = {"flag_if": "if_score", "flag_lof": "lof_score"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SHORT = {"Isolation Forest": "Isolation Forest", "Local Outlier Factor": "LOF", "Z-score (per home & hour)": "Z-score",
         "IQR (per home & hour)": "IQR", "Global z-score (naive)": "Global z-score", "Consensus (2+ methods)": "2+ agree"}

st.set_page_config(page_title="Pakistan Electricity Anomaly Mining", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
_card_ids = itertools.count()          # restarts every run, so card keys stay stable


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


# ---------------------------------------------------------------- building blocks
def header(step, title, sub):
    st.markdown(f'<div class="eyebrow">{step}</div><h1 class="page-title">{title}</h1><p class="page-sub">{sub}</p>',
                unsafe_allow_html=True)


def card():
    return st.container(key=f"card_{next(_card_ids)}")


def insight(title, sub=""):
    st.markdown(f'<div class="insight">{title}</div><div class="insight-sub">{sub}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", hot=False):
    col.markdown(f'<div class="kpi{" hot" if hot else ""}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def callout(html, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{html}</div>', unsafe_allow_html=True)


def gap(px_height=12):
    """Small vertical spacer, halved for the compact layout."""
    st.markdown(f"<div style='height:{max(2, px_height // 2)}px'></div>", unsafe_allow_html=True)


def style(fig, ytitle=None, xtitle=None, height=360, hover="x"):
    """One chart look everywhere: hairline solid grid, recessive axes, 2px lines, legend top-left."""
    fig.update_layout(
        height=height, margin=dict(l=4, r=8, t=28, b=8), template="plotly_white", font=dict(family=FONT, size=13, color=INK2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode=hover, hoverdistance=30,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#c3c2b7", font=dict(family=FONT, size=13, color="#0b0b0b")),
        legend=dict(orientation="h", x=0, y=1.14, xanchor="left", font=dict(size=13)),
        xaxis=dict(title=xtitle, showgrid=False, linecolor=BASELINE, ticks="outside", tickcolor=BASELINE, title_font=dict(size=13)),
        yaxis=dict(title=ytitle, gridcolor=GRID, gridwidth=1, zeroline=False, title_font=dict(size=13)))
    return fig


def heatmap(d):
    """Scorecard heatmap: light-to-dark blue cells, white text on dark cells and near-black on light ones."""
    z = d.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(z=z, x=list(d.columns), y=list(d.index), colorscale=SEQUENTIAL, zmin=0, zmax=1, showscale=False,
                               xgap=2, ygap=2, hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    for i, row in enumerate(z):
        for j, v in enumerate(row):
            if v == v:      # skip NaN
                fig.add_annotation(x=d.columns[j], y=d.index[i], text=f"{v:.2f}", showarrow=False,
                                   font=dict(size=13, color="#ffffff" if v >= 0.6 else "#0b0b0b"))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def chart_card(title, sub, fig, how=None, table=None):
    """A chart in a card: insight title, the chart, an optional how-to-read line and a table view."""
    with card():
        insight(title, sub)
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        if how:
            st.markdown(f'<div class="how"><b>How to read:</b> {how}</div>', unsafe_allow_html=True)
        if table is not None:
            with st.expander("Table view"):
                st.dataframe(table, width="stretch", hide_index=True)


def rnd(df, n=2):
    """Round only the numeric columns (DataFrame.round warns when a table also has dates)."""
    return df.assign(**{c: df[c].round(n) for c in df.select_dtypes("number").columns})


def empty_msg(msg="No data for this selection. Try another household or a wider date range."):
    st.info(msg)


def house_picker(key, label="Household"):
    s = load_csv("household_summary.csv")
    labels = {f"House {r.household_id} ({r.city})": int(r.household_id) for r in s.itertuples()}
    return labels[st.selectbox(label, list(labels), key=key)]


def go_to(page):
    st.session_state["nav"] = page


def footer_nav(current):
    names = list(PAGES)
    i = names.index(current)
    gap(18)
    left, right = st.columns(2)
    if i > 0:
        left.button(f"← {names[i - 1]}", on_click=go_to, args=(names[i - 1],), key=f"back_{i}")
    if i < len(names) - 1:
        right.button(f"Next: {names[i + 1]} →", on_click=go_to, args=(names[i + 1],), type="primary", key=f"next_{i}")


def ts(t, fmt="%d %b %Y, %H:%M"):
    return pd.Timestamp(t).strftime(fmt)


# ---------------------------------------------------------------- pages
def page_home():
    df = load_hourly()
    comp = load_csv("method_comparison.csv").set_index("method")
    wc = load_csv("weather_check.csv")
    header("Data Mining project · 2026", "Finding unusual electricity use in Pakistani homes",
           "We learn what is <b>normal for each of 59 households</b>, allow for the weather, and flag the hours that do not fit. "
           "Built on real one-minute smart-meter data from six cities.")
    st.markdown("".join(f'<span class="chip">{n}</span>' for n in ["Izaan Imran", "Fahad Ahmed", "Shiraz Asif", "Dayyan Hashmi"]),
                unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], "Hours analysed", f"{len(df):,}", f"{ts(df.timestamp.min(), '%b %Y')} to {ts(df.timestamp.max(), '%b %Y')}")
    kpi(c[1], "Flagged by 2+ methods", f"{int(df.consensus2.sum()):,}", "high-confidence unusual hours", hot=True)
    kpi(c[2], "Flagged by Isolation Forest", f"{int(df.flag_if.sum()):,}", "1% of hours by design")
    kpi(c[3], "Households", f"{df.household_id.nunique()}", f"across {df.city.nunique()} cities in Pakistan")
    gap(14)

    left, right = st.columns([3, 2])
    with left:
        cases = load_csv("cases.csv")
        ctx = load_parquet("case_context.parquet")
        case = cases.iloc[0]
        w = ctx[ctx.case_id == case.case_id].sort_values("timestamp")
        a = w[w.consensus > 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.kwh, mode="lines", line=dict(color=NORMAL, width=2), name="Energy used",
                                 hovertemplate="%{y:.2f} kWh<extra></extra>"))
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.base_med, mode="lines", line=dict(color=MUTED, width=1.5), name="Normal for this hour",
                                 hovertemplate="%{y:.2f} kWh<extra></extra>"))
        fig.add_trace(go.Scatter(x=a.timestamp, y=a.kwh, mode="markers", name="Flagged",
                                 marker=dict(color=ANOMALY, size=9, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="%{y:.2f} kWh (flagged)<extra></extra>"))
        chart_card(f"A real example: House {case.household_id} ({case.city})",
                   f"Around {ts(case.timestamp, '%d %b %Y')} the home used far more than its normal for that hour. Orange marks the flagged hours.",
                   style(fig, "Energy (kWh per hour)", height=330),
                   how="the blue line is real use, the grey line is what is normal for that hour of day. Orange dots are hours our methods flagged.")
    with right:
        with card():
            insight("How it works", "Four ideas, in plain words")
            st.markdown(
                '<div class="step"><div class="num">1</div><div class="txt"><b>Learn normal, per home</b>Typical use for every hour, weekday or weekend, for each household.</div></div>'
                '<div class="step"><div class="num">2</div><div class="txt"><b>Allow for the weather</b>Hot hours raise cooling use, so a high reading in a heatwave is not automatically unusual.</div></div>'
                '<div class="step"><div class="num">3</div><div class="txt"><b>Flag what does not fit</b>Isolation Forest, LOF, Z-score and IQR each vote. Agreement means higher confidence.</div></div>'
                '<div class="step"><div class="num">4</div><div class="txt"><b>Explain every flag</b>Each flagged hour gets a plain-language reason.</div></div>',
                unsafe_allow_html=True)
        with card():
            insight("Where to look", "A suggested path for the demo")
            st.markdown(
                '<div class="where"><b>Anomaly Detector</b> pick any home, see its flagged hours and the reason for each.</div>'
                '<div class="where"><b>Weather Impact</b> see which high readings are explained by heat.</div>'
                '<div class="where"><b>Method Comparison</b> see which method to trust, and for what.</div>',
                unsafe_allow_html=True)
    gap(14)

    t_hot = wc[wc.measure.str.contains("time-only")].iloc[0]
    w_hot = wc[wc.measure.str.contains("weather-aware")].iloc[0]
    cons = comp.loc["Consensus (2+ methods)"]
    best = comp["auc"].idxmax()
    f = st.columns(3)
    for col, big, label in [
        (f[0], f"{t_hot.ratio_hot_vs_other:.1f}× → {w_hot.ratio_hot_vs_other:.1f}×", "How much more often the hottest 10% of hours look “high”, before and after allowing for temperature."),
        (f[1], f"{cons.precision:.2f}", "Precision when two or more methods agree (injection test). Use these as high-confidence alerts."),
        (f[2], f"{comp.loc[best, 'auc']:.2f}", f"Best overall ranking quality (AUC): {SHORT.get(best, best)}. No single method wins every anomaly type.")]:
        with col:
            with card():
                st.markdown(f'<div class="pill">Key finding</div><div class="fig">{big}</div><div class="fig-label">{label}</div>', unsafe_allow_html=True)
    gap(6)
    callout(DISCLAIMER, warn=True)
    st.button("Open the Anomaly Detector →", on_click=go_to, args=("Anomaly Detector",), type="primary", key="cta_home")
    footer_nav("Home")


def page_explorer():
    header("Step 1 of 7 · Explore", "Data Explorer",
           "When do homes use electricity, how does it change across the year, and how does it relate to the weather?")
    df = load_hourly()
    with card():
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
    gap(10)
    t1, t2, t3, t4 = st.tabs(["Daily pattern", "Over time", "Seasons", "Weather link"])
    with t1:
        prof = d.groupby(["hour", "is_weekend"])["kwh"].mean().reset_index()
        prof["Day type"] = prof["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        allh = d.groupby("hour")["kwh"].mean()
        fig = px.line(prof, x="hour", y="kwh", color="Day type", color_discrete_map={"Weekday": NORMAL, "Weekend": MUTED})
        fig.update_traces(line=dict(width=2), hovertemplate="%{y:.2f} kWh")
        chart_card(f"Use peaks around {allh.idxmax():02d}:00 and is lowest around {allh.idxmin():02d}:00",
                   "Average energy per hour of the day for the selection.",
                   style(fig, "Average energy (kWh per hour)", "Hour of day", 380),
                   how="each point is the average energy used in that hour. Higher points are the busy hours (peaks).",
                   table=prof[["hour", "Day type", "kwh"]].pipe(rnd, 3).rename(columns={"kwh": "Average kWh"}))
    with t2:
        day = d.assign(date=d.timestamp.dt.floor("D")).groupby("date")["kwh"].mean().reset_index()
        fig = px.line(day, x="date", y="kwh", color_discrete_sequence=[NORMAL])
        fig.update_traces(line=dict(width=2), hovertemplate="%{y:.2f} kWh")
        chart_card("How daily use changes over time",
                   f"One point per day. Daily average ranges from {day.kwh.min():.2f} to {day.kwh.max():.2f} kWh per hour.",
                   style(fig, "Average energy (kWh per hour)", None, 380),
                   how="bumps in summer usually mean cooling.", table=day.pipe(rnd, 3).rename(columns={"kwh": "Average kWh"}))
    with t3:
        mon = d.groupby("month")["kwh"].mean().reindex(range(1, 13))
        busy, quiet = int(mon.idxmax()), int(mon.idxmin())
        mdf = pd.DataFrame({"Month": MONTHS, "kwh": mon.values})
        fig = go.Figure(go.Bar(x=mdf.Month, y=mdf.kwh, marker=dict(color=NORMAL, cornerradius=4), width=0.55,
                               hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>"))
        chart_card(f"{MONTHS[busy - 1]} is the busiest month, {mon.max() / mon.min():.1f}× the quietest ({MONTHS[quiet - 1]})",
                   "Average energy per hour for each calendar month. Months with no bar have no data for this selection.",
                   style(fig, "Average energy (kWh per hour)", None, 360, hover="closest"),
                   table=mdf.pipe(rnd, 3).rename(columns={"kwh": "Average kWh"}))
    with t4:
        corr = load_csv("weather_corr.csv")
        corr = corr if city == "All cities" else corr[corr.city == city]
        r = float(corr["temp"].mean())
        s = d.sample(min(len(d), 6000), random_state=42)
        fig = px.scatter(s, x="temp_c", y="kwh", opacity=0.35, color_discrete_sequence=[NORMAL])
        fig.update_traces(marker=dict(size=7), hovertemplate="%{x:.0f} °C: %{y:.2f} kWh<extra></extra>")
        title = "Hotter hours go with higher use" if r > 0.2 else "Only a weak link between temperature and use"
        chart_card(f"{title} (correlation {r:.2f})",
                   f"Each dot is one hour. Average correlation for this selection: temperature {r:.2f}, humidity {corr['humidity'].mean():.2f}, rain {corr['precip'].mean():.2f} (closer to 1 is a stronger link).",
                   style(fig, "Energy (kWh per hour)", "Outdoor temperature (°C)", 380, hover="closest"),
                   how="if the dots rise to the right, the home uses more energy when it is hot.",
                   table=s[["timestamp", "temp_c", "kwh"]].head(500).pipe(rnd, 2))
    footer_nav("Data Explorer")


def page_normal():
    header("Step 2 of 7 · Normal", "Normal Behaviour",
           "What a typical day looks like for one home. This is the baseline every hour is compared with.")
    with card():
        hid = house_picker("normal_house")
    prof = load_csv("profile_hourly.csv")
    p = prof[prof.household_id == hid]
    if p.empty:
        empty_msg(); return
    wk = p[p.is_weekend == 0].sort_values("hour")
    we = p[p.is_weekend == 1].sort_values("hour")
    gap(10)
    callout(f"On weekdays this home typically peaks around <b>{int(wk.loc[wk.normal.idxmax(), 'hour']):02d}:00</b> "
            f"({wk.normal.max():.2f} kWh) and is quietest around <b>{int(wk.loc[wk.normal.idxmin(), 'hour']):02d}:00</b> ({wk.normal.min():.2f} kWh). "
            "The shaded band holds the middle 80% of days; a reading far outside it is unusual for this home.")
    ymax = float(max(wk.p90.max(), we.p90.max())) * 1.05          # same scale on both panels so they can be compared
    cols = st.columns(2)
    for col, q, title in [(cols[0], wk, "Weekdays"), (cols[1], we, "Weekends")]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q.hour, y=q.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.hour, y=q.p10, fill="tonexty", fillcolor="rgba(42,120,214,0.14)", line=dict(width=0),
                                 name="Normal band (10%–90%)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.hour, y=q.normal, line=dict(color=NORMAL, width=2), name="Typical (median)",
                                 hovertemplate="%{y:.2f} kWh<extra></extra>"))
        with col:
            chart_card(title, "Typical use for each hour of the day (both panels share one scale)",
                       style(fig, "Energy (kWh per hour)", "Hour of day", 340).update_yaxes(range=[0, ymax]),
                       table=q[["hour", "p10", "normal", "p90"]].rename(columns={"p10": "Low (10%)", "normal": "Typical", "p90": "High (90%)"}))
    footer_nav("Normal Behaviour")


def page_detector():
    header("Step 3 of 7 · Detect", "Anomaly Detector",
           "Pick a home and a method to see which hours were flagged, how strongly, and why.")
    callout(DISCLAIMER, warn=True)
    df = load_hourly()
    with card():
        c1, c2, c3 = st.columns([1, 1.2, 1.6])
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
        rng = c3.date_input("Date range (starts around this home's strongest anomaly)", (start, end), min_value=lo, max_value=hi)
    if len(rng) != 2:
        empty_msg("Pick both a start and an end date."); return
    w = h[(h.timestamp.dt.date >= rng[0]) & (h.timestamp.dt.date <= rng[1])]
    if w.empty:
        empty_msg(); return
    a = w[w[flag]]
    gap(10)
    k = st.columns(4)
    kpi(k[0], "Hours shown", f"{len(w):,}", f"{rng[0]:%d %b} to {rng[1]:%d %b %Y}")
    kpi(k[1], "Flagged hours", f"{len(a):,}", method_name.split(" (")[0], hot=True)
    kpi(k[2], "Share flagged", f"{100 * len(a) / len(w):.1f}%", "of the hours shown")
    if len(a):
        s = a.loc[a.kwh.sub(a.base_med).idxmax()]
        kpi(k[3], "Strongest flagged hour", f"{s.kwh:.2f} kWh", f"{ts(s.timestamp)} · normal {s.base_med:.2f}")
    else:
        kpi(k[3], "Strongest flagged hour", "none", "no hours flagged in this window")
    gap(12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.kwh, mode="lines", line=dict(color=NORMAL, width=2), name="Energy used",
                             hovertemplate="%{y:.2f} kWh<extra>Used</extra>"))
    fig.add_trace(go.Scatter(x=w.timestamp, y=w.base_med, mode="lines", line=dict(color=MUTED, width=1.5), name="Normal for this hour",
                             hovertemplate="%{y:.2f} kWh<extra>Normal</extra>"))
    if len(a):
        fig.add_trace(go.Scatter(
            x=a.timestamp, y=a.kwh, mode="markers", name="Flagged (bigger = more methods agree)",
            marker=dict(color=ANOMALY, size=8 + 3 * a.consensus.clip(lower=1), line=dict(color=SURFACE, width=2)),
            customdata=a[["reason", "base_med", "consensus"]].to_numpy(),
            hovertemplate="<b>%{y:.2f} kWh</b> (normal %{customdata[1]:.2f})<br>%{customdata[0]}<br>%{customdata[2]} of 4 methods agree<extra>Flagged</extra>"))
        s = a.loc[a.kwh.sub(a.base_med).idxmax()]
        fig.add_annotation(x=s.timestamp, y=s.kwh, text="Strongest", showarrow=True, arrowhead=0, ax=0, ay=-34,
                           font=dict(size=12, color=INK2), arrowcolor=BASELINE)
    title = (f"House {hid}: {len(a)} flagged hours in {len(w):,} ({100 * len(a) / len(w):.1f}%)" if len(a)
             else f"House {hid}: no hours flagged in this window")
    chart_card(title, f"Method: {method_name}", style(fig, "Energy (kWh per hour)", None, 420, hover="closest"),
               how="blue is real use, grey is what is normal for that hour. Orange dots are flagged hours; hover one to see why.",
               table=w[["timestamp", "kwh", "base_med", "reason", "consensus"]].pipe(rnd, 2).rename(columns={
                   "kwh": "Used (kWh)", "base_med": "Normal (kWh)", "reason": "Reason", "consensus": "Methods agreeing"}).head(1000))

    if flag in SCORES:
        fig2 = go.Figure(go.Scatter(x=w.timestamp, y=w[SCORES[flag]], mode="lines", line=dict(color=MUTED, width=1.5), name="Score",
                                    hovertemplate="%{y:.2f}<extra>Score</extra>"))
        if len(a):
            fig2.add_trace(go.Scatter(x=a.timestamp, y=a[SCORES[flag]], mode="markers", name="Flagged",
                                      marker=dict(color=ANOMALY, size=9, line=dict(color=SURFACE, width=2))))
        gap(12)
        chart_card("Anomaly score", "The model's score for each hour. Flagged hours are the highest scores.",
                   style(fig2, "Score (higher = more unusual)", None, 240, hover="closest"))

    gap(12)
    t1, t2 = st.tabs(["Flagged hours", "Whole days that look unusual"])
    with t1:
        if a.empty:
            st.info("No hours were flagged in this window. Try a wider date range or another method.")
        else:
            tab = a[["timestamp", "kwh", "base_med", "temp_c", "reason", "consensus"]].rename(columns={
                "timestamp": "Time", "kwh": "Used (kWh)", "base_med": "Normal (kWh)", "temp_c": "Temp (°C)",
                "reason": "Why it was flagged", "consensus": "Methods agreeing (of 4)"})
            st.dataframe(tab.pipe(rnd, 2), width="stretch", hide_index=True)
            st.download_button("Download flagged hours (CSV)", tab.to_csv(index=False).encode(), f"house_{hid}_flagged.csv", "text/csv")
    with t2:
        day = load_parquet("daily_anomalies.parquet")
        day = day[(day.household_id == hid) & day.flag_daily]
        if day.empty:
            st.info("No unusual days for this household.")
        else:
            st.dataframe(day[["date", "total_kwh", "peak_kwh", "daily_reason"]].rename(columns={
                "date": "Day", "total_kwh": "Total (kWh/day)", "peak_kwh": "Highest hour (kWh)", "daily_reason": "Why"}).pipe(rnd, 2),
                width="stretch", hide_index=True)
        st.caption("A day is flagged when its total or shape is unusual for this home (Isolation Forest on daily features).")
    footer_nav("Anomaly Detector")


def page_clusters():
    header("Step 4 of 7 · Group", "Household Clusters",
           "Households grouped by how they use electricity: size, time of day, weekends and sensitivity to heat.")
    hc = load_csv("household_clusters.csv")
    summ = load_csv("cluster_summary.csv")
    var = load_csv("pca_variance.csv")
    names = sorted(hc.cluster_name.unique())
    color = {n: CLUSTER_COLORS[i % 3] for i, n in enumerate(names)}
    symbol = {n: CLUSTER_SYMBOLS[i % 3] for i, n in enumerate(names)}
    fig = go.Figure()
    for n in names:
        g = hc[hc.cluster_name == n]
        fig.add_trace(go.Scatter(x=g.pc1, y=g.pc2, mode="markers", name=n.replace(": ", " · "),
                                 marker=dict(color=color[n], symbol=symbol[n], size=12, line=dict(color=SURFACE, width=2)),
                                 customdata=g[["household_id", "city", "mean_kwh"]].to_numpy(),
                                 hovertemplate="House %{customdata[0]} (%{customdata[1]})<br>%{customdata[2]:.2f} kWh per hour<extra></extra>"))
    style(fig, f"PC2 ({100 * var.explained[1]:.0f}% of variation)", f"PC1 ({100 * var.explained[0]:.0f}% of variation)", 470, hover="closest")
    fig.update_layout(legend=dict(orientation="h", x=0, y=-0.18, xanchor="left"))
    chart_card(f"{len(hc)} homes fall into {len(names)} behaviour groups",
               "Each dot is a household. Dots close together behave alike. Colour and shape both show the group.",
               fig, how="PCA squeezes 8 behaviour numbers into 2 axes just so we can draw them. Groups overlap, so read them as tendencies, not sharp boxes.",
               table=hc[["household_id", "city", "cluster_name", "mean_kwh"]].rename(columns={"cluster_name": "Group", "mean_kwh": "Average kWh/h"}).pipe(rnd, 2))
    gap(12)
    cols = st.columns(len(summ))
    for col, n in zip(cols, names):
        r = summ[summ.cluster_name == n].iloc[0]
        cities = hc[hc.cluster_name == n].city.value_counts()
        with col:
            with card():
                st.markdown(f'<div class="pill"><span class="swatch" style="background:{color[n]}"></span>{n.split(":")[0]}</div>'
                            f'<div class="insight">{n.split(": ", 1)[1].capitalize()}</div>'
                            f'<div class="insight-sub">{int(r.households)} homes · {r.mean_kwh:.2f} kWh per hour on average<br>'
                            f'{r.evening_share:.0%} of use in the evening · {r.night_share:.0%} at night<br>'
                            f'Link to temperature {r.temp_corr:.2f} · flagged by 2+ methods {r.multi_method_rate_pct:.1f}%<br>'
                            f'Mostly in: {", ".join(cities.index[:3])}</div>', unsafe_allow_html=True)
    gap(8)
    with st.expander("How the number of groups (K) was chosen"):
        k = load_csv("cluster_k_selection.csv")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k.k, y=k.silhouette, name="Silhouette (higher is better)", mode="lines+markers",
                                 line=dict(color=NORMAL, width=2), marker=dict(size=9, line=dict(color=SURFACE, width=2))))
        st.plotly_chart(style(fig, "Silhouette", "Number of groups K", 280, hover="x"), width="stretch", config=PLOT_CONFIG)
        st.caption(f"The silhouette score is low (best {k.silhouette.max():.2f}), so groups overlap. We chose K = {len(names)}, the best score "
                   "among K = 3 to 6, because K = 2 is too coarse to be useful. Isolation Forest flags about 1% of hours in every home by design, "
                   "so groups are compared by how often two or more methods agree.")
        st.dataframe(k.pipe(rnd, 3), width="stretch", hide_index=True)
    footer_nav("Household Clusters")


def page_weather():
    header("Step 5 of 7 · Weather", "Weather Impact",
           "Hot weather raises cooling use, so a high reading is not always unusual. Here we separate heat-driven highs from truly unusual ones.")
    df = load_hourly()
    wc = load_csv("weather_check.csv")
    t_hot = wc[wc.measure.str.contains("time-only")].iloc[0]
    w_hot = wc[wc.measure.str.contains("weather-aware")].iloc[0]
    callout(f"Without weather, the hottest 10% of hours look “high” <b>{t_hot.ratio_hot_vs_other:.1f}×</b> more often than other hours. "
            f"After allowing for temperature this drops to <b>{w_hot.ratio_hot_vs_other:.1f}×</b>, so most heat-driven highs are explained.")
    t1, t2, t3 = st.tabs(["Energy vs temperature", "Explained vs truly unusual", "Heat check"])
    with t1:
        with card():
            hid = house_picker("wx_house")
        gap(10)
        h = df[df.household_id == hid]
        s = h.sample(min(len(h), 6000), random_state=42)
        fl = s[s.flag_if]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s[~s.flag_if].temp_c, y=s[~s.flag_if].kwh, mode="markers", name="Normal",
                                 marker=dict(color=NORMAL, size=6, opacity=0.35), hovertemplate="%{x:.0f} °C: %{y:.2f} kWh<extra></extra>"))
        fig.add_trace(go.Scatter(x=fl.temp_c, y=fl.kwh, mode="markers", name="Flagged (Isolation Forest)",
                                 marker=dict(color=ANOMALY, size=9, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="%{x:.0f} °C: %{y:.2f} kWh (flagged)<extra></extra>"))
        chart_card(f"House {hid}: where do the flagged hours sit?",
                   "Flagged hours inside the rising cloud of dots (hot and busy) are less surprising than those far above it.",
                   style(fig, "Energy (kWh per hour)", "Outdoor temperature (°C)", 380, hover="closest"),
                   how="each dot is an hour. Orange dots are flagged hours.", table=s[["timestamp", "temp_c", "kwh", "flag_if"]].head(500).pipe(rnd, 2))
        tr = load_csv("temp_response.csv")
        tr = tr[tr.household_id == hid]
        if not tr.empty:
            gap(12)
            f2 = go.Figure(go.Scatter(x=tr.temp_bin, y=tr.mean_kwh, mode="lines+markers", line=dict(color=NORMAL, width=2),
                                      marker=dict(size=9, line=dict(color=SURFACE, width=2)), hovertemplate="%{x} °C: %{y:.2f} kWh<extra></extra>"))
            chart_card("How this home's use responds to temperature", "Average energy in each 2-degree temperature band.",
                       style(f2, "Average kWh per hour", "Temperature band (°C)", 280), table=tr[["temp_bin", "mean_kwh", "hours"]])
    with t2:
        f = df[df.consensus > 0]
        rc = f.reason.value_counts().reset_index()
        rc.columns = ["Reason", "Hours"]
        rc = rc.iloc[::-1]
        n_exp = int((f.reason == "High for this hour, but typical for this temperature").sum())
        n_unu = int((f.reason == "Much higher than usual for this hour, even allowing for temperature").sum())
        fig = go.Figure(go.Bar(x=rc.Hours, y=rc.Reason, orientation="h", marker=dict(color=NORMAL, cornerradius=4), width=0.5,
                               hovertemplate="%{y}: %{x:,} hours<extra></extra>"))
        style(fig, None, "Flagged hours", 80 + 50 * len(rc), hover="closest").update_layout(margin=dict(l=4, r=24, t=10, b=8))
        chart_card(f"{n_unu:,} high hours stay unusual even after allowing for temperature; {n_exp:,} are typical for the heat",
                   "Why hours flagged by at least one method were flagged. Every flagged hour gets one plain-language reason.",
                   fig, how="“Typical for this temperature” means the weather-aware baseline does not find the hour unusual.", table=rc.iloc[::-1])
    with t3:
        fig = go.Figure()
        hv = wc[wc.measure.str.startswith("High")]
        labels = ["Time-only baseline", "Weather-aware baseline"]
        fig.add_trace(go.Bar(x=labels, y=hv.hot_hours_pct, name="Hottest 10% of hours", marker=dict(color=ANOMALY, cornerradius=4),
                             hovertemplate="%{y:.1f}% of hot hours<extra></extra>"))
        fig.add_trace(go.Bar(x=labels, y=hv.other_hours_pct, name="All other hours", marker=dict(color=NORMAL, cornerradius=4),
                             hovertemplate="%{y:.1f}% of other hours<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.55, bargroupgap=0.08)
        chart_card("Allowing for temperature removes most of the heat effect",
                   "Share of hours called “high”, for the hottest 10% of hours versus all other hours.",
                   style(fig, "Hours called “high” (%)", None, 340, hover="closest"),
                   how="the orange bar is the hottest 10% of hours. The closer it is to the blue bar, the less the baseline is fooled by heat.",
                   table=wc.rename(columns={"measure": "Measure", "hot_hours_pct": "Hottest 10% (%)", "other_hours_pct": "Other hours (%)", "ratio_hot_vs_other": "Ratio"}))
        st.caption("The model-based flags (Isolation Forest, Z-score, IQR) still cluster in heat, so some extreme-heat hours remain flagged. See the table view.")
    footer_nav("Weather Impact")


def page_methods():
    header("Step 6 of 7 · Evaluate", "Method Comparison & Evaluation",
           "There are no true anomaly labels in this dataset, so we tested each method by injecting known anomalies into real data. "
           "This is a controlled test, not real labels.")
    comp = load_csv("method_comparison.csv")
    inj = load_csv("injection_results.csv")
    cons = comp[comp.method.str.startswith("Consensus")].iloc[0]
    lof = comp[comp.method == "Local Outlier Factor"].iloc[0]
    spike_auc = inj[(inj.method == "Isolation Forest") & (inj.type == "spike")].auc.iloc[0]
    st.markdown("**Which method should you use?**")
    r = st.columns(3)
    for col, pill, title, text in [
        (r[0], "1 · Rank", "Isolation Forest", f"Best at spikes (AUC {spike_auc:.2f}), fast, gives a score to rank hours. Our main method."),
        (r[1], "2 · Confirm", "2 or more methods agree", f"Precision {cons.precision:.2f}: when several methods agree it is almost always a real injected anomaly. Use for high-confidence alerts."),
        (r[2], "3 · Look for stuck values", "Local Outlier Factor", f"Best overall ranking (AUC {lof.auc:.2f}) and the only method that notices flat-lined readings, but only {lof.precision:.2f} precision.")]:
        with col:
            with card():
                st.markdown(f'<div class="pill">{pill}</div><div class="insight">{title}</div><div class="insight-sub">{text}</div>', unsafe_allow_html=True)
    gap(14)

    with card():
        metric = st.radio("Score to show", ["auc", "recall", "f1"], horizontal=True,
                          format_func=lambda m: {"auc": "AUC (ranking quality)", "recall": "Recall (found)", "f1": "F1 (balance)"}[m])
    order = ["spike", "drop", "shift", "flatline", "all"]
    d = inj[inj.type.isin(order)].pivot(index="method", columns="type", values=metric)[order]
    d = d.reindex(list(SHORT)).rename(index=SHORT, columns={"all": "Overall", "spike": "Spike", "drop": "Drop to zero", "shift": "6-hour shift", "flatline": "Flat line"})
    fig = heatmap(d)
    style(fig, None, None, 330, hover="closest")
    fig.update_xaxes(tickfont=dict(size=13))
    fig.update_yaxes(tickfont=dict(size=13))
    gap(10)
    chart_card("Scorecard: how well each method finds each kind of injected anomaly",
               "Darker blue is better. Spike = one very high hour; Drop = sudden zero; 6-hour shift = use ×2.5; Flat line = value stuck for 6 hours.",
               fig, how="read across a row to see a method's strengths. No row is strong everywhere, which is why we combine methods.",
               table=d.pipe(rnd, 3).reset_index().rename(columns={"method": "Method"}))
    gap(12)
    cc = st.columns(2)
    with cc[0]:
        agr = load_csv("agreement.csv")
        nm = {"flag_if": "Isolation Forest", "flag_lof": "LOF", "flag_z": "Z-score", "flag_iqr": "IQR"}
        pv = agr.pivot(index="method_a", columns="method_b", values="jaccard").rename(index=nm, columns=nm)
        pv = pv.loc[list(nm.values()), list(nm.values())]
        fig = heatmap(pv)
        style(fig, None, None, 330, hover="closest")
        chart_card("The methods catch different things", "Overlap of flagged hours (1.0 = identical, 0 = none).",
                   fig, how="low numbers off the diagonal mean the methods disagree, so combining them adds information.", table=pv.pipe(rnd, 3).reset_index())
    with cc[1]:
        with card():
            insight("Summary table", "Flag rate on real data and injection-test scores")
            st.dataframe(comp.rename(columns={"method": "Method", "flag_rate_pct": "Flags on real data (%)", "precision": "Precision",
                         "recall": "Recall", "f1": "F1", "auc": "AUC"}), width="stretch", hide_index=True)
            st.caption("Precision: of the hours flagged, how many were injected. Recall: of the injected hours, how many were found. "
                       "AUC: how well the raw score ranks injected hours above normal ones, whatever the threshold.")
            sens = load_csv("sensitivity.csv")
            st.markdown("**Sensitivity (contamination).** Isolation Forest flags grow in step with the expected share: " +
                        ", ".join(f"{x.contamination:.1%} → {int(x.flagged_hours):,} hours" for x in sens.itertuples()) + ".")
    gap(8)
    callout(f"The naive <b>global z-score</b> (all homes together) is the weakest method (F1 {comp[comp.method.str.startswith('Global')].f1.iloc[0]:.2f}). "
            "That is why we learn what is normal <b>per household</b>.")
    footer_nav("Method Comparison")


def page_about():
    header("Step 7 of 7 · Context", "About & Limitations", "Where the data comes from, what we did, and what you should keep in mind.")
    c = st.columns(2)
    with c[0]:
        with card():
            insight("Data")
            st.markdown("**Residential Energy and Weather Data Pakistan (REWD-P)**, LUMS Energy Institute: "
                        "[lei.lums.edu.pk](https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html). "
                        "Smart-meter readings every minute from households in Lahore, Multan, Peshawar, Karachi, Islamabad and Skardu, plus hourly weather.")
        gap(12)
        with card():
            insight("What we did, in simple words")
            st.markdown(
                '<div class="step"><div class="num">1</div><div class="txt">Cleaned the minute data and turned it into <b style="display:inline">hourly energy</b> (kWh).</div></div>'
                '<div class="step"><div class="num">2</div><div class="txt">Added the <b style="display:inline">weather</b> of the home\'s city for each hour.</div></div>'
                '<div class="step"><div class="num">3</div><div class="txt">Learned each home\'s <b style="display:inline">normal pattern</b>: typical use per hour, weekday or weekend, and temperature range.</div></div>'
                '<div class="step"><div class="num">4</div><div class="txt">Measured how far each hour is from normal and flagged unusual hours with four methods.</div></div>'
                '<div class="step"><div class="num">5</div><div class="txt">Grouped similar homes with K-Means and tested the methods by injecting known anomalies.</div></div>',
                unsafe_allow_html=True)
    with c[1]:
        with card():
            insight("Limitations")
            st.markdown(
                "- **No real anomaly labels.** Accuracy comes from injected, synthetic anomalies; real ones may differ.\n"
                "- **59 households, not 60** as the proposal says (the dataset page lists 59). Coverage differs by home and some have gaps.\n"
                "- **No timezone** in the source. We assume local Pakistan time.\n"
                "- **Solar and net metering:** two homes report generation and some may use net metering. We use the 'Usage' column as given.\n"
                "- **Zero readings** may be real outages (load-shedding). We keep them and flag them as 'near zero'.\n"
                "- **Offline analysis:** each home's normal is learned from the whole recorded period, not in real time.\n"
                "- **Modest cluster separation** (silhouette about 0.2) and only 59 homes, so results describe this sample, not all of Pakistan.")
    gap(6)
    callout(DISCLAIMER, warn=True)
    footer_nav("About & Limitations")


PAGES = {"Home": page_home, "Data Explorer": page_explorer, "Normal Behaviour": page_normal,
         "Anomaly Detector": page_detector, "Household Clusters": page_clusters, "Weather Impact": page_weather,
         "Method Comparison": page_methods, "About & Limitations": page_about}

if "nav" not in st.session_state:      # open any page directly with ?page=Anomaly%20Detector
    st.session_state["nav"] = st.query_params.get("page") if st.query_params.get("page") in PAGES else "Home"
with st.sidebar:
    st.markdown('<div class="brand-row"><div class="logo">⚡</div><div><div class="brand">Anomaly Mining</div>'
                '<div class="brand-sub">Pakistan household electricity</div></div></div>'
                '<div class="side-label">The story</div>', unsafe_allow_html=True)
    choice = st.radio("Go to", list(PAGES), key="nav", label_visibility="collapsed")
    step, n_steps = list(PAGES).index(choice), len(PAGES) - 1          # Home is the start, the other 7 pages are steps 1 to 7
    st.markdown(f'<div class="side-label">Progress · {"start" if step == 0 else f"step {step} of {n_steps}"}</div>'
                f'<div class="prog"><span style="width:{100 * step / n_steps:.0f}%"></span></div>'
                '<div class="side-label" style="margin-top:12px">Key</div>'
                '<span class="side-chip"><span class="swatch" style="background:var(--blue)"></span>Normal</span>'
                '<span class="side-chip"><span class="swatch" style="background:var(--orange)"></span>Flagged</span>'
                '<div class="side-label" style="margin-top:12px">Data</div>'
                '<span class="side-chip">REWD-P · LUMS</span><span class="side-chip">59 homes · 6 cities</span>'
                '<div class="side-note" style="margin-top:12px"><b>Theme:</b> ⋮ menu (top right) → Settings → Theme '
                'for light or dark.</div>', unsafe_allow_html=True)
st.query_params["page"] = choice
PAGES[choice]()
