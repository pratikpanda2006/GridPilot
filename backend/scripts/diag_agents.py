import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from app import data
from app.agents.peer import PeerAgent
from app.agents.trend import TrendAgent
from app.config import DEFAULT_SETTINGS

sd = data.load_site("Robin")
peer = PeerAgent(DEFAULT_SETTINGS)
peer._groups = peer._build_groups(sd)
print("groups:", {k: len(v) for k, v in peer._groups.items()})

t0 = sd.meter["timestamp"].min()
now = t0 + pd.Timedelta(hours=4000)
gn = {}
for kind, members in peer._groups.items():
    if len(members) < 2:
        continue
    vals = []
    for b in members:
        v = peer.latest_norm(sd, b, now)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) >= 2:
        gn[kind] = np.array(vals)

zs = []
for building in sd.buildings:
    for k, members in peer._groups.items():
        if building in members and k in gn:
            val = peer.latest_norm(sd, building, now)
            if not np.isfinite(val):
                continue
            z = (val - gn[k].mean()) / gn[k].std()
            zs.append((building, k, round(z,2)))
zs.sort(key=lambda t: abs(t[2]), reverse=True)
print("max |z|:", zs[:10])
print("frac |z|>2:", np.mean([abs(z[2])>2 for z in zs]) if zs else 0)

# trend check
tr = TrendAgent(DEFAULT_SETTINGS)
t = t0 + pd.Timedelta(hours=5000)
for b in sd.buildings[:8]:
    s = tr.window_norm(sd, b, t, 168).dropna()
    if s.shape[0] >= 24:
        x = np.arange(s.shape[0]); y = s.values
        slope = np.polyfit(x,y,1)[0]
        print(b, "n",s.shape[0], "slope", round(float(slope),5))
