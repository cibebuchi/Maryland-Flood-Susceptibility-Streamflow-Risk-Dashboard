import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from fetch_data import (
    get_sites_in_county,
    fetch_streamflow,
    compute_susceptibility,
    get_risk_level,
    detect_flood_events
)

# === Paths & Config ===
BASE = Path(__file__).parent
logo_path = BASE / "logo.png"

st.set_page_config(
    page_title="Maryland Flood Susceptibility",
    page_icon=str(logo_path),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit default menu, header, and footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar branding & navigation
st.sidebar.image(str(logo_path), use_container_width=True)
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Select View", ["About", "Analyze Stations", "Completeness Analysis", "Visualize Risk Map"]
)

# === Data Loading ===
counties = gpd.read_file(BASE / "counties.geojson")

# === Helpers ===
def colored_risk(level):
    if level == "Low":
        return f"<span style='color:green;font-weight:bold;font-size:24px'>{level}</span>"
    elif level == "Moderate":
        return f"<span style='color:#FF9933;font-weight:bold;font-size:24px'>{level}</span>"
    else:
        return f"<span style='color:#CC0000;font-weight:bold;font-size:24px'>{level}</span>"


def colored_overflow(flag):
    if flag:
        return f"<span style='color:#CC0000;font-weight:bold;font-size:24px'>True</span>"
    else:
        return f"<span style='color:green;font-weight:bold;font-size:24px'>False</span>"

# === Sections ===

def display_about():
    st.markdown(
        """
        ### About the App

        This app uses a pretrained explainable AI model designed to assess flood 
        susceptibility of U.S. Geological Survey (USGS) flow stations in Maryland. 
        The model analyzes key features of flow stations to identify flooding risk.

        **Risk categories:**
        - **Low Risk:** 0–33%
        - **Moderate Risk:** 33–66%
        - **High Risk:** >66%

        **Overflow criteria:**
        1. Discharge >95th percentile for ≥2 consecutive days, or
        2. Discharge >99th percentile on ≥2 days total

        Data source: **U.S. Geological Survey (USGS)** real-time monitoring.
        """
    )


def display_completeness():
    st.header("USGS Station Completeness Analysis")
    img_path = BASE / "MDUSGS.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.error("Completeness image not found.")


def display_risk_map():
    st.header("Maryland Flood Risk Map")
    img_path = BASE / "Riskmap.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
        st.markdown(
            "Risk Scores assigned per USGS county using data from 1970–2025 with up to 95% completeness."
        )
    else:
        st.error("Risk map image not found.")


def analyze_flood_risk():
    st.header("Flood Susceptibility & Overflow Analysis")
    county = st.selectbox("Select Maryland County", sorted(counties['NAME_2'].unique()))
    selected = counties[counties['NAME_2'] == county]
    stations_df = get_sites_in_county(selected)

    if stations_df.empty:
        st.warning("No USGS stations found in this county.")
        return

    stations_df['label'] = stations_df['site_no'] + ' – ' + stations_df['station_nm']
    label = st.selectbox("Select Station", stations_df['label'])
    row = stations_df[stations_df['label'] == label].iloc[0]

    if st.button("Fetch & Analyze"):
        df = fetch_streamflow(row['site_no'])
        if df['discharge_cfs'].dropna().empty:
            st.error("No discharge data available.")
            return

        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        recent = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=7)]

        score = compute_susceptibility(df, row['dec_long_va'], row['dec_lat_va'])
        level = get_risk_level(score)

        p95, p99 = np.percentile(df['discharge_cfs'].dropna(), [95, 99])
        events = detect_flood_events(df, p95)
        overflow = any(e[0] for e in events if df['Date'].iloc[e[0]] >= recent['Date'].min())
        overflow |= (recent['discharge_cfs'] > p99).sum() >= 2

        st.metric("Susceptibility Score", f"{score:.3f}")
        st.markdown(f"**Risk Level:** {colored_risk(level)}", unsafe_allow_html=True)
        st.markdown(f"**Recent Overflow:** {colored_overflow(overflow)}", unsafe_allow_html=True)
        st.write(f"Data range: {recent['Date'].min().date()} to {recent['Date'].max().date()}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent['Date'], y=recent['discharge_cfs'], mode='lines+markers', name='Discharge'
        ))
        fig.add_hline(y=p95, line_dash='dash', annotation_text='95th %ile')
        fig.add_hline(y=p99, line_dash='dashdot', annotation_text='99th %ile')
        fig.update_layout(title='Discharge (Last 7 Days)', xaxis_title='Date', yaxis_title='cfs')
        st.plotly_chart(fig, use_container_width=True)

# === Main ===
if mode == "About":
    display_about()
elif mode == "Analyze Stations":
    analyze_flood_risk()
elif mode == "Completeness Analysis":
    display_completeness()
else:
    display_risk_map()

# Footer branding
st.markdown("<hr><p style='text-align:center;color:#888;'>Created by Chibuike Ibebuchi</p>", unsafe_allow_html=True)
