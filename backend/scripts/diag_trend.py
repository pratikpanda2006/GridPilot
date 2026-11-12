import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from app import data
from app.agents.base import linear_fit
from app.config import DEFAULT_SETTINGS

sd = data.load_site("Robin")
t0 = sd.meter["timestamp"].min()
window_h = int(DEFAULT_SETTINGS["trend_window_hours"])
rows = []
for off in range(1500, 8760*2, 400):
    now = t0 + pd.Timedelta(hours=off)
    for b in sd.buildings:
        s = sd.building_norm_series(b)
        sub = s[(s.index <= now) & (s.index >= now - pd.Timedelta(hours=window_h))].dropna()
        if sub.shape[0] < 24:
            continue
        x = np.arange(sub.shape[0]); y = sub.values
        slope, ic, r2 = linear_fit(x, y)
        diffs = np.diff(y)
        sign_frac = float(np.mean(np.sign(diffs) == np.sign(slope))) if slope != 0 else 0.0
        ml = float(np.nanmean(y))
        norm_slope = abs(slope)/ml if ml > 0 else 0
        rows.append((norm_slope, r2, sign_frac))
df = pd.DataFrame(rows, columns=["norm_slope","r2","sign_frac"])
print("rows", len(df))
print(df.describe(percentiles=[.5,.75,.9,.95,.99]).round(4))
print("count norm_slope>=0.0004 & r2>=0.25 & sign_frac>=0.6:", int(((df.norm_slope>=0.0004)&(df.r2>=0.25)&(df.sign_frac>=0.6)).sum()))
print("count norm_slope>=0.0002:", int((df.norm_slope>=0.0002).sum()))
