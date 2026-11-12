import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from app import data
from app.agents import build_agents
from app.config import DEFAULT_SETTINGS
from app.orchestrator import Orchestrator

sd = data.load_site("Robin")
settings = dict(DEFAULT_SETTINGS)
agents = build_agents(settings)
orch = Orchestrator(settings)

t0 = sd.meter["timestamp"].min()

def run(now):
    all_msgs = []
    for a in agents:
        all_msgs.extend(agents[a].emit(sd, now))
    candidates, stats = orch.process(sd, now, all_msgs)
    return all_msgs, candidates, stats

# 1) normal run at a time where messages exist
now = t0 + pd.Timedelta(hours=5000)
msgs, cands, stats = run(now)
print("== STEP at", now, "==")
print("agents messages:", {a: len(agents[a].emit(sd, now)) for a in agents})
print("orchestrator stats:", stats)
print("candidates:", len(cands))
if cands:
    c = cands[0]
    print("sample candidate building:", c["building"])
    print("sample candidate features subset:", {k: c["features"][k] for k in list(c["features"])[:8]})
    print("sample candidate feature count:", len(c["features"]))

# 2) empty messages
empty_cands, empty_stats = orch.process(sd, now, [])
print("== EMPTY messages stats:", empty_stats, "candidates:", len(empty_cands))

# 3) invalid building id
bad = agents["fixed_limit"].emit(sd, now)
if bad:
    bad_msg = bad[0]
    bad_msg.__dict__["building"] = "Robin_nonexistent_XYZ"
    bad_cands, bad_stats = orch.process(sd, now, bad_msg if isinstance(bad_msg, list) else [bad_msg])
    print("== INVALID building stats:", bad_stats)

# 4) full-timeline agent counts (sampled)
counts = {a: 0 for a in agents}
for h in range(1500, 8760*2, 200):
    n = t0 + pd.Timedelta(hours=h)
    for a in agents:
        counts[a] += len(agents[a].emit(sd, n))
print("== sampled agent totals:", counts)
