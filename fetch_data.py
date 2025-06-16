import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import dataretrieval.nwis as nwis
from pathlib import Path
from utils import extract_features, detect_flood_events, extract_dem_features

# Load trained model & config
BASE = Path(__file__).parent
MODEL = joblib.load(BASE / 'model.joblib')
with open(BASE / 'config.json') as f:
    CFG = json.load(f)


def get_sites_in_county(county_gdf):
    """
    Returns USGS stations within the given Maryland county polygon GeoDataFrame.
    """
    # Focus exclusively on Maryland
    geom = county_gdf.geometry.iloc[0]
    PARAM = '00060'

    # Query MD stations only
    info_df, _ = nwis.get_info(
        stateCd='MD',
        parameterCd=PARAM,
        siteTypeCd='ST'
    )
    if info_df.empty:
        return pd.DataFrame(columns=info_df.columns)

    pts = gpd.GeoDataFrame(
        info_df,
        geometry=gpd.points_from_xy(info_df.dec_long_va, info_df.dec_lat_va),
        crs='EPSG:4326'
    )
    inside = pts[pts.within(geom)]
    return inside.reset_index(drop=True)


def fetch_streamflow(site_id):
    """
    Fetch daily discharge from 2000-01-01 to today, drop timezone info.
    """
    start = '2000-01-01'
    end = pd.Timestamp.now().strftime('%Y-%m-%d')
    df, _ = nwis.get_dv(
        sites=site_id,
        parameterCd='00060',
        start=start,
        end=end
    )
    # Drop timezone from index
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename_axis('Date').reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Rename discharge column
    col = next((c for c in df.columns if '00060' in c or 'discharge' in c.lower()), None)
    if col:
        df = df.rename(columns={col: 'discharge_cfs'})[['Date', 'discharge_cfs']]
    else:
        df['discharge_cfs'] = np.nan
    return df


def compute_susceptibility(df, lon, lat):
    """
    Compute continuous flood susceptibility (0–1) using trained model.
    """
    feats = extract_features(df, lon, lat)
    sel = CFG.get('selected_indices', [])
    X_sel = feats[sel].reshape(1, -1)
    if hasattr(MODEL, 'predict_proba'):
        return float(MODEL.predict_proba(X_sel)[0,1])
    else:
        return float(MODEL.predict(X_sel)[0])


def get_risk_level(score):
    """
    Simple 3-tier risk: Low <0.33, Moderate <0.66, High >=0.66
    """
    if score < 0.33:
        return 'Low'
    elif score < 0.66:
        return 'Moderate'
    else:
        return 'High'


def recent_overflow_risk(df):
    """
    Return True if any 2-day flood event (>95th percentile) happened
    within the last CFG['event_days_window'] days in Maryland.
    """
    window = CFG.get('event_days_window', 7)
    if df['discharge_cfs'].notna().any():
        thresh = np.percentile(df['discharge_cfs'].dropna(), 95)
        events = detect_flood_events(df, thresh)
    else:
        events = []

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=window)
    cutoff = cutoff.tz_localize(None) if hasattr(cutoff, 'tz') else cutoff

    for start, _ in events:
        d = df['Date'].iloc[start]
        d = d.tz_localize(None) if hasattr(d, 'tz') else d
        if d >= cutoff:
            return True
    return False
