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

# Streamlit configuration (with custom page icon)
logo_path = Path(__file__).parent / "logo.jpg"
st.set_page_config(
    page_title="Maryland Flood Susceptibility",
    page_icon=str(logo_path),
    layout="wide"
)
st.title("Maryland Flood Susceptibility & Streamflow Risk Dashboard")

# Load MD counties GeoJSON
BASE     = Path(__file__).parent
counties = gpd.read_file(BASE / "counties.geojson")

# Sidebar mode selector
def display_mode():
    mode = st.sidebar.radio(
        "Mode", ["About", "Analyze Stations", "Completeness Analysis", "Visualize Risk Map"]
    )
    return mode

# Helper to color-code risk level text with larger size
def colored_risk(level):
    if level == "Low":
        return f"<span style='color:green;font-weight:bold;font-size:24px'>{level}</span>"
    elif level == "Moderate":
        return f"<span style='color:#FF9933;font-weight:bold;font-size:24px'>{level}</span>"
    else:
        return f"<span style='color:#CC0000;font-weight:bold;font-size:24px'>{level}</span>"

# Helper to color-code boolean overflow flag with larger size
def colored_overflow(flag):
    if flag:
        return f"<span style='color:#CC0000;font-weight:bold;font-size:24px'>True</span>"
    else:
        return f"<span style='color:green;font-weight:bold;font-size:24px'>False</span>"

# About Section
def display_about():
    st.markdown("""
    ### About the App

    This app uses a pretrained explainable AI model designed to assess flood 
    susceptibility of U.S. Geological Survey (USGS) flow stations in Maryland. 
    The model has been trained to analyze key features of flow stations, 
    specifically focusing on identifying conditions that contribute to flooding risk.

    The model uses historical discharge data to assign flood susceptibility scores, categorizing stations 
    into three risk levels:
    - **Low Risk** stations represent susceptibility scores between 0% and 33%.
    - **Moderate Risk** stations fall between 33% and 66%.
    - **High Risk** stations show scores exceeding 66%.

    Flood overflow risk is determined by analyzing discharge values, particularly when they either:
    1. Exceed the 95th percentile for at least two consecutive days over the past 7 days, **or**
    2. Exceed the 99th percentile on at least two total days (not necessarily consecutive) over the past 7 days.

    All data is sourced from the **U.S. Geological Survey (USGS)**, ensuring reliable, real‐time monitoring 
    of flow conditions and flood susceptibility.
    """)

# Completeness Analysis Section
def display_completeness():
    st.header("USGS Station Completeness Analysis")
    img_path = BASE / "MDUSGS.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.error("Completeness image not found.")

# Visualize Risk Map Section
def display_risk_map():
    st.header("Maryland Flood Risk Map")
    img_path = BASE / "Riskmap.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
        st.markdown(
            "Risk Scores were assigned at each USGS county with sufficient data using the pretrained AI model, "
            "covering data from 1970 to 2025 with up to 95% completeness records."
        )
    else:
        st.error("Risk map image not found.")

# Analyze flood susceptibility and overflow risk
def analyze_flood_risk():
    # County selector (Maryland only)
    county_list = sorted(counties['NAME_2'].unique())
    county      = st.selectbox("Select Maryland County", county_list)

    # Fetch stations in selected county
    selected_county = counties[counties['NAME_2'] == county]
    stations_df     = get_sites_in_county(selected_county)

    if stations_df.empty:
        st.warning("No USGS stations found in this county.")
        return

    # Station selector
    stations_df["label"] = stations_df["site_no"] + " – " + stations_df["station_nm"]
    station_label        = st.selectbox("Select Station", stations_df["label"].tolist())
    station_row          = stations_df[stations_df["label"] == station_label].iloc[0]

    if st.button("Fetch & Analyze"):
        df = fetch_streamflow(station_row["site_no"])
        if df["discharge_cfs"].dropna().empty:
            st.error("No discharge data available.")
            return

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date")
        max_date = df["Date"].max()
        start_date = max_date - pd.Timedelta(days=7)
        recent_df  = df[(df["Date"] >= start_date) & (df["Date"] <= max_date)]

        score = compute_susceptibility(df, station_row["dec_long_va"], station_row["dec_lat_va"])
        level = get_risk_level(score)

        thresh_95 = np.percentile(df["discharge_cfs"].dropna(), 95)
        thresh_99 = np.percentile(df["discharge_cfs"].dropna(), 99)
        events_95 = detect_flood_events(df, thresh_95)
        overflow_95 = any(e[0] for e in events_95 if df["Date"].iloc[e[0]] >= start_date)
        count_99 = (recent_df["discharge_cfs"] > thresh_99).sum()
        overflow_99 = count_99 >= 2
        overflow = overflow_95 or overflow_99

        st.metric("Flood Susceptibility Score", f"{score:.3f}")
        st.markdown(f"**Risk Level:** {colored_risk(level)}", unsafe_allow_html=True)
        st.markdown(f"**Recent Potential Overflow:** {colored_overflow(overflow)}", unsafe_allow_html=True)
        st.write(f"Data range: {start_date.date()} to {max_date.date()}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_df["Date"],
            y=recent_df["discharge_cfs"],
            mode="lines+markers",
            name="Discharge",
            line=dict(width=4, color="blue"),  # bold blue line
            marker=dict(size=6, color="blue")  # blue markers
        ))
        # faint grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        # percentile lines
        fig.add_hline(y=thresh_95, line_dash="dash", line_color="red",
                      annotation_text="95th Percentile", annotation_font_color="black")
        fig.add_hline(y=thresh_99, line_dash="dashdot", line_color="darkred",
                      annotation_text="99th Percentile", annotation_font_color="black")
        fig.update_layout(
            title="Discharge (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title="Discharge (cfs)",
            plot_bgcolor="white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Main
mode = display_mode()
if mode == "About":
    display_about()
elif mode == "Analyze Stations":
    analyze_flood_risk()
elif mode == "Completeness Analysis":
    display_completeness()
else:
    display_risk_map()

# Footer
st.markdown("<hr><p style='text-align:center;color:#888;'>Created by Chibuike Ibebuchi</p>", unsafe_allow_html=True)
