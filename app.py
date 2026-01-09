import streamlit as st
import streamlit.components.v1 as components

import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path

# Local project imports (must exist in your repo)
from fetch_data import (
    get_sites_in_county,
    fetch_streamflow,
    compute_susceptibility,
    get_risk_level,
    detect_flood_events,
)

# ============================================================
# Paths & Config
# ============================================================
BASE = Path(__file__).parent

LOGO_PATH = BASE / "logo.png"
COUNTIES_GEOJSON = BASE / "counties.geojson"

# Files you mentioned (place them in the repo root next to app.py)
RISK_MAP_HTML = BASE / "interactive_risk_map.html"   # <- put your HTML here
FLOOD_SUMMARY_XLSX = BASE / "Flood summary.xlsx"     # <- put your Excel here

st.set_page_config(
    page_title="Maryland Flood Risk Dashboard",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default menu, header, and footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar
# ============================================================
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Assess potential stream overflow in real time",
        "Historical Flood Risk Assessment",
        "About",
    ],
)

# ============================================================
# Data Loading
# ============================================================
@st.cache_data(show_spinner=False)
def load_counties():
    if not COUNTIES_GEOJSON.exists():
        return None
    return gpd.read_file(COUNTIES_GEOJSON)

@st.cache_data(show_spinner=False)
def load_excel(path: Path) -> dict[str, pd.DataFrame]:
    """Load all sheets from an Excel file into a dict."""
    xls = pd.ExcelFile(path)
    out: dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        out[sheet] = df
    return out

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column-name variations to stable names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ren = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"county", "county_name", "county_na", "county nam", "county name"}:
            ren[c] = "county_name"
        elif lc in {"totalcbgs", "total_cbgs", "total cbgs"}:
            ren[c] = "total_cbgs"
        elif lc in {"high_risk_cbgs", "high risk cbgs", "high_risk_cbg"}:
            ren[c] = "high_risk_cbgs"
        elif lc in {"high_risk_pct", "high risk pct", "high_risk_p", "high_risk_%", "high_risk_percent"}:
            ren[c] = "high_risk_pct"
        elif "total_home" in lc and "value" in lc:
            ren[c] = "total_home_value"
        elif lc in {"total_population", "total_popu", "total_pop", "population"}:
            ren[c] = "total_population"
        elif ("high" in lc and "home" in lc and "value" in lc) or lc in {"high_risk_home_value"}:
            ren[c] = "high_risk_home_value"
        elif ("high" in lc and "pop" in lc) or lc in {"high_risk_population"}:
            ren[c] = "high_risk_population"
        elif lc in {"risk_level", "risk level", "flood_risk_level", "flood risk level"}:
            ren[c] = "risk_level"
    df = df.rename(columns=ren)
    return df

counties_gdf = load_counties()

# ============================================================
# Formatting helpers
# ============================================================
def colored_risk(level: str) -> str:
    if level == "Low":
        return f"<span style='color:green;font-weight:bold;font-size:22px'>{level}</span>"
    if level == "Moderate":
        return f"<span style='color:#FF9933;font-weight:bold;font-size:22px'>{level}</span>"
    return f"<span style='color:#CC0000;font-weight:bold;font-size:22px'>{level}</span>"

def colored_overflow(flag: bool) -> str:
    if flag:
        return "<span style='color:#CC0000;font-weight:bold;font-size:22px'>True</span>"
    return "<span style='color:green;font-weight:bold;font-size:22px'>False</span>"

def fmt_compact(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}K"
    if ax.is_integer():
        return f"{int(x)}"
    return f"{x:.2f}"

# ============================================================
# Pages / Sections
# ============================================================
def display_general_about():
    st.title("Maryland Flood Risk Dashboard")
    st.markdown(
        """
This app provides **real-time** and **historical** flood-risk assessment capabilities for Maryland.

- **Real-time assessment** evaluates USGS streamflow stations to flag **potential stream overflow** based on recent discharge behavior.
- **Historical assessment** provides **modelled flood susceptibility** using XAI, plus **population and real estate exposure** and a **social vulnerability composite flood risk** at the Census Block Group (CBG) level.

**Method reference**  
Ibebuchi C.C. (2026) *Equity-focused flood risk assessment in Maryland using a hybrid explainable machine learning framework with FEMA and USGS data.* **Natural Hazards.** DOI: 10.1007/s11069-025-07923-8
        """
    )

def display_method():
    st.subheader("Method (Real-time station assessment)")
    st.markdown(
        """
This real-time module uses a pretrained explainable AI model to assess the flood susceptibility of
U.S. Geological Survey (USGS) flow stations in Maryland and flags **recent overflow** conditions.

**Risk categories**
- **Low Risk:** 0–33%
- **Moderate Risk:** 33–66%
- **High Risk:** >66%

**Overflow criteria (recent window)**
1) Discharge >95th percentile for ≥2 consecutive days, **or**  
2) Discharge >99th percentile on ≥2 days total

Data source: **U.S. Geological Survey (USGS)** real-time monitoring.
        """
    )

def analyze_stations():
    st.subheader("Analyze USGS Stations (Real-time)")

    if counties_gdf is None or counties_gdf.empty:
        st.error(
            "Could not load counties. Ensure `counties.geojson` exists in your repository root next to `app.py`."
        )
        return

    county = st.selectbox("Select Maryland County", sorted(counties_gdf["NAME_2"].unique()))
    selected = counties_gdf[counties_gdf["NAME_2"] == county]
    stations_df = get_sites_in_county(selected)

    if stations_df.empty:
        st.warning("No USGS stations found in this county.")
        return

    stations_df = stations_df.copy()
    stations_df["label"] = stations_df["site_no"].astype(str) + " – " + stations_df["station_nm"].astype(str)
    label = st.selectbox("Select Station", stations_df["label"])
    row = stations_df.loc[stations_df["label"] == label].iloc[0]

    st.caption("Tip: Click **Fetch & Analyze** to compute susceptibility and check recent overflow conditions.")
    if st.button("Fetch & Analyze", type="primary"):
        df = fetch_streamflow(row["site_no"])
        if df is None or df.empty or df.get("discharge_cfs") is None or df["discharge_cfs"].dropna().empty:
            st.error("No discharge data available for this station.")
            return

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Recent window for overflow detection
        recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=7)].copy()
        if recent.empty:
            st.error("Recent window is empty (no data in the last 7 days).")
            return

        # Modelled susceptibility
        score = compute_susceptibility(df, row["dec_long_va"], row["dec_lat_va"])
        level = get_risk_level(score)

        # Percentile thresholds from the full available record
        discharge = df["discharge_cfs"].dropna().to_numpy()
        p95, p99 = np.percentile(discharge, [95, 99])

        # Flood-event detection using your existing helper
        events = detect_flood_events(df, p95)

        # Was there any detected event that starts inside the recent window?
        overflow = any(
            (idx0 is not None) and (df["Date"].iloc[int(idx0)] >= recent["Date"].min())
            for (idx0, *_rest) in events
        )

        # Additional criterion: >=2 days above 99th percentile in recent window
        overflow = bool(overflow) or ((recent["discharge_cfs"] > p99).sum() >= 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Susceptibility Score", f"{score:.3f}")
        c2.markdown(f"**Risk Level:** {colored_risk(level)}", unsafe_allow_html=True)
        c3.markdown(f"**Recent Overflow:** {colored_overflow(overflow)}", unsafe_allow_html=True)

        st.write(f"Recent window: {recent['Date'].min().date()} to {recent['Date'].max().date()}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent["Date"],
                y=recent["discharge_cfs"],
                mode="lines+markers",
                name="Discharge (cfs)",
            )
        )
        fig.add_hline(y=float(p95), line_dash="dash", annotation_text="95th %ile")
        fig.add_hline(y=float(p99), line_dash="dashdot", annotation_text="99th %ile")
        fig.update_layout(
            title="Discharge (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title="cfs",
        )
        st.plotly_chart(fig, use_container_width=True)

def historical_risk_map():
    st.subheader("Interactive Flood Risk Map (Historical Assessment)")

    if RISK_MAP_HTML.exists():
        html = RISK_MAP_HTML.read_text(encoding="utf-8", errors="ignore")
        components.html(html, height=780, scrolling=True)
        st.caption("If the map does not render correctly, ensure the HTML is self-contained (no missing local JS/CSS files).")
    else:
        st.warning(
            "Could not find `interactive_risk_map.html` in the repository root. "
            "Add it next to `app.py`, or upload it below for a quick test."
        )
        upl = st.file_uploader("Upload interactive_risk_map.html", type=["html"])
        if upl is not None:
            html = upl.read().decode("utf-8", errors="ignore")
            components.html(html, height=780, scrolling=True)

def county_summary():
    st.subheader("County Summary (Historical Assessment)")

    # Load from repo if available, else allow upload
    data_book = None
    if FLOOD_SUMMARY_XLSX.exists():
        try:
            data_book = load_excel(FLOOD_SUMMARY_XLSX)
        except Exception as e:
            st.error(f"Failed to read `{FLOOD_SUMMARY_XLSX.name}`. Error: {e}")
            data_book = None

    if data_book is None:
        st.warning(
            "Could not find `Flood summary.xlsx` in the repository root. "
            "Add it next to `app.py`, or upload it below for a quick test."
        )
        upl = st.file_uploader("Upload Flood summary.xlsx", type=["xlsx"])
        if upl is None:
            return
        try:
            data_book = load_excel(Path(upl.name))  # dummy path for cache key
        except Exception:
            # Fallback: read directly from bytes
            data_book = {"Uploaded": pd.read_excel(upl)}

    # Choose a sheet (if multiple)
    sheet_names = list(data_book.keys())
    sheet = st.selectbox("Select Excel sheet", sheet_names, index=0)
    df_raw = data_book[sheet]
    df = _normalize_cols(df_raw)

    st.write("Data preview")
    st.dataframe(df.head(50), use_container_width=True)

    metric_map = {}

    if "high_risk_pct" in df.columns:
        metric_map["High-risk CBGs (%)"] = ("high_risk_pct", "Percent of CBGs in high-risk class")
    if "high_risk_cbgs" in df.columns:
        metric_map["High-risk CBG count"] = ("high_risk_cbgs", "Number of high-risk CBGs")
    if "total_home_value" in df.columns:
        metric_map["Total home value ($)"] = ("total_home_value", "Total home value (as provided in the sheet)")
    if "total_population" in df.columns:
        metric_map["Total population"] = ("total_population", "Total population (as provided in the sheet)")
    if "high_risk_home_value" in df.columns:
        metric_map["Home value in high-risk CBGs ($)"] = ("high_risk_home_value", "Home value within high-risk CBGs")
    if "high_risk_population" in df.columns:
        metric_map["Population in high-risk CBGs"] = ("high_risk_population", "Population within high-risk CBGs")

    has_risk_level = "risk_level" in df.columns

    if not metric_map:
        st.info(
            "I couldn't detect the expected columns in this sheet. "
            "Expected columns include: county_name, high_risk_pct, high_risk_cbgs, total_home_value, total_population."
        )
        return

    st.markdown("#### Interactive plotting")

    metric_label = st.selectbox("Select metric to plot", list(metric_map.keys()))
    metric_col, metric_desc = metric_map[metric_label]
    st.caption(metric_desc)

    if "county_name" in df.columns:
        x_mode = st.radio("X-axis", ["County", "Top-N counties"], horizontal=True)
        plot_df = df.dropna(subset=["county_name", metric_col]).copy()
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric_col])

        if x_mode == "Top-N counties":
            n = st.slider("N", 5, 24, 10)
            plot_df = plot_df.sort_values(metric_col, ascending=False).head(n)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=plot_df["county_name"],
                y=plot_df[metric_col],
                text=[fmt_compact(v) for v in plot_df[metric_col]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"{metric_label} by County",
            xaxis_title="County",
            yaxis_title=metric_label,
            xaxis_tickangle=-35,
            margin=dict(t=70, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif has_risk_level:
        plot_df = df.dropna(subset=["risk_level", metric_col]).copy()
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric_col])

        try:
            plot_df["_risk_num"] = pd.to_numeric(plot_df["risk_level"], errors="coerce")
            if plot_df["_risk_num"].notna().any():
                plot_df = plot_df.sort_values("_risk_num")
        except Exception:
            pass

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=plot_df["risk_level"].astype(str),
                y=plot_df[metric_col],
                text=[fmt_compact(v) for v in plot_df[metric_col]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"{metric_label} by Flood Risk Level",
            xaxis_title="Flood Risk Level",
            yaxis_title=metric_label,
            margin=dict(t=70, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("This sheet doesn't look like a county table or a risk-level table (missing `county_name` or `risk_level`).")

    st.markdown(
        """
**Note on interpretation**  
The app plots the columns exactly as provided in the Excel file. If you intend “total_home_value” and “total_population”
to represent *only* high-risk CBGs (exposure in high-risk areas), ensure the sheet is prepared that way or include explicit
columns like `high_risk_home_value` and `high_risk_population`.
        """
    )

# ============================================================
# Render
# ============================================================
if section == "About":
    display_general_about()

elif section == "Assess potential stream overflow in real time":
    st.title("Assess potential stream overflow in real time")
    tab_method, tab_stations = st.tabs(["Method", "Analyze Stations"])
    with tab_method:
        display_method()
    with tab_stations:
        analyze_stations()

else:
    st.title("Historical Flood Risk Assessment")
    tab_map, tab_county = st.tabs(["Flood Risk Map", "County Summary"])
    with tab_map:
        historical_risk_map()
    with tab_county:
        county_summary()

st.markdown("<hr><p style='text-align:center;color:#888;'>Created by Chibuike Ibebuchi</p>", unsafe_allow_html=True)
