import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from app import data
from app.agents.base import linear_fit

sd = data.load_site("Robin")
t0 = sd.meter["timestamp"].min()
rows = []
for off in range(1500, 8760*2, 400):
    now = t0 + pd.Timedelta(hours=off)
    for b in sd.buildings:
        sub = sd.building_norm_series(b)
        sub = sub[(sub.index <= now) & (sub.index >= now - pd.Timedelta(days=12))].dropna()
        if sub.shape[0] < 24*5:
            continue
        daily = sub.resample("24h").mean().dropna()
        if daily.shape[0] < 5:
            continue
        x = np.arange(daily.shape[0]); y = daily.values
        slope, ic, r2 = linear_fit(x, y)
        diffs = np.diff(y)
        sign_frac = float(np.mean(np.sign(diffs) == np.sign(slope))) if slope != 0 else 0.0
        ml = float(np.nanmean(y))
        norm_daily = abs(slope)/ml if ml > 0 else 0
        rows.append((norm_daily, r2, sign_frac))
df = pd.DataFrame(rows, columns=["norm_daily","r2","sign_frac"])
print("rows", len(df))
print(df.describe(percentiles=[.5,.75,.9,.95]).round(4))
print("count norm_daily>=0.008 & r2>=0.4 & sign_frac>=0.6:", int(((df.norm_daily>=0.008)&(df.r2>=0.4)&(df.sign_frac>=0.6)).sum()))
print("count norm_daily>=0.005 & r2>=0.35:", int(((df.norm_daily>=0.005)&(df.r2>=0.35)).sum()))
