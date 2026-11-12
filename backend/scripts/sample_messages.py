import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from app import data
from app.agents import build_agents
from app.config import DEFAULT_SETTINGS

sd = data.load_site("Robin")
agents = build_agents(dict(DEFAULT_SETTINGS))
t0 = sd.meter["timestamp"].min()
samples = {}
for h in range(1500, 8760 * 2, 24):
    now = t0 + pd.Timedelta(hours=h)
    for a in agents:
        if a in samples:
            continue
        for m in agents[a].emit(sd, now):
            samples[a] = m
            break
print("== one complete message per agent ==")
for a in ["fixed_limit", "weather_adjusted", "peer", "trend"]:
    m = samples.get(a)
    print(f"\n--- {a} ---")
    if m is None:
        print("  (no message captured)")
        continue
    for k, v in m.to_dict().items():
        print(f"  {k}: {v}")
