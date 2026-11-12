import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from app import data
from app.agents import build_agents
from app.config import DEFAULT_SETTINGS

sd = data.load_site("Robin")
ag = build_agents(DEFAULT_SETTINGS)
t0 = sd.meter["timestamp"].min()
counts = {k: 0 for k in ag}
first = {}
for h in range(500, 8760*2, 150):
    now = t0 + pd.Timedelta(hours=h)
    for a in ag:
        msgs = ag[a].emit(sd, now)
        if msgs:
            counts[a] += len(msgs)
            first.setdefault(a, (h, msgs[0].message_type, msgs[0].building))
print("offs by agent (sampled every 48h):")
for a in ag:
    print(f"  {a}: {counts[a]} messages, first@hour {first.get(a)}")
