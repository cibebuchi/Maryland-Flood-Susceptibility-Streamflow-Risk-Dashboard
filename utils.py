"""
Utilities for Maryland flood susceptibility modeling:
- DEM feature extraction (elevation, slope) using Maryland-specific DEM
- Flood event detection and feature engineering on USGS discharge data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform
from scipy.ndimage import sobel

# DEM file containing Maryland elevation mask
BASE = Path(__file__).parent
DEM_PATH = BASE / 'USA1_msk_alt.tif'  # DEM file is directly in FloodApp/  # Maryland-only DEM

if not DEM_PATH.exists():
    raise FileNotFoundError(f"Maryland DEM not found at {DEM_PATH}")

# Load DEM on import
_dem_ds = rasterio.open(str(DEM_PATH))
_dem_array = _dem_ds.read(1, masked=True)
_dem_transform = _dem_ds.transform
_dem_crs = _dem_ds.crs
_dem_nodata = _dem_ds.nodata
_dem_res = _dem_ds.res


def extract_dem_features(lon, lat):
    """
    Given a longitude/latitude in Maryland, return elevation and slope
    from the Maryland DEM. Returns (elev, slope) in meters and dimensionless.
    """
    xs, ys = transform('EPSG:4326', _dem_crs, [lon], [lat])
    row, col = rowcol(_dem_transform, xs[0], ys[0])
    if not (0 <= row < _dem_array.shape[0] and 0 <= col < _dem_array.shape[1]):
        return np.nan, np.nan
    val = _dem_array[row, col]
    if np.ma.is_masked(val) or (_dem_nodata is not None and np.isclose(val, _dem_nodata)):
        return np.nan, np.nan
    elev = float(val)
    # compute slope (rise/run) using Sobel filter on 3x3 window
    if 0 < row < _dem_array.shape[0]-1 and 0 < col < _dem_array.shape[1]-1:
        win = _dem_array[row-1:row+2, col-1:col+2]
        if not win.mask.any():
            dx = sobel(win, axis=1) / (8 * _dem_res[0] * 111320)
            dy = sobel(win, axis=0) / (8 * _dem_res[1] * 111320)
            slope = float(np.hypot(dx, dy).mean())
        else:
            slope = np.nan
    else:
        slope = np.nan
    return elev, slope


def detect_flood_events(df, threshold):
    """
    Identify flood events where discharge exceeds threshold for 2+ consecutive days.
    Returns list of (start_index, end_index) tuples.
    """
    events = []
    ex = df['discharge_cfs'] > threshold
    i, n = 0, len(ex)
    while i < n-1:
        if ex.iloc[i] and ex.iloc[i+1]:
            start = i
            while i < n and ex.iloc[i]:
                i += 1
            events.append((start, i-1))
        else:
            i += 1
    return events


def extract_features(df, lon, lat):
    """
    Compute feature vector for a Maryland station:
    [mean daily diff, elev, slope, seasonal medians (4), P95 (4), P5 (4)]
    Impute missing using seasonal medians.
    """
    d = df.copy()
    d['Month'] = d['Date'].dt.month
    seasons = {
        'Winter': [12,1,2], 'Spring': [3,4,5],
        'Summer': [6,7,8], 'Fall': [9,10,11]
    }
    med, p95, p5 = {}, {}, {}
    for name, months in seasons.items():
        vals = d[d['Month'].isin(months)]['discharge_cfs'].dropna()
        med[name] = vals.median() if not vals.empty else d['discharge_cfs'].median()
        p95[name] = np.percentile(vals,95) if not vals.empty else np.percentile(d['discharge_cfs'].dropna(),95)
        p5[name]  = np.percentile(vals,5)  if not vals.empty else np.percentile(d['discharge_cfs'].dropna(),5)

    # Impute missing discharge by seasonal medians
    d['discharge_cfs'] = d.apply(
        lambda r: med[[s for s,ms in seasons.items() if r['Month'] in ms][0]]
                  if pd.isna(r['discharge_cfs']) else r['discharge_cfs'],
        axis=1
    )

    # Basic trends
    diff = d['discharge_cfs'].diff().mean() or 0.0
    d['Year'] = d['Date'].dt.year
    annual = d.groupby('Year')['discharge_cfs'].mean()
    slope = np.polyfit(np.arange(len(annual)), annual.values,1)[0] if len(annual)>1 else 0.0

    # DEM features
    elev, slp = extract_dem_features(lon, lat)

    # Assemble feature vector
    feats = [diff, elev, slp]
    feats += [med[s] for s in ['Winter','Spring','Summer','Fall']]
    feats += [p95[s] for s in ['Winter','Spring','Summer','Fall']]
    feats += [p5[s]  for s in ['Winter','Spring','Summer','Fall']]

    return np.array(feats, dtype=float)
