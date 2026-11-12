"""Feature engineering for the intelligence gate.

All features must be computable from information available *before* the
decision time (no future leakage). Building attributes (type, size) and the
agent message group provide the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGENT_KEYS = ["fixed_limit", "weather_adjusted", "peer", "trend"]
TYPE_DUMMIES = ["education", "lodging", "office", "public", "assembly", "retail", "other"]


def building_type_code(primary: str) -> str:
    if not isinstance(primary, str):
        return "other"
    p = primary.lower()
    if "education" in p:
        return "education"
    if "lodging" in p or "residential" in p:
        return "lodging"
    if "office" in p:
        return "office"
    if "public" in p:
        return "public"
    if "assembly" in p or "entertainment" in p:
        return "assembly"
    if "retail" in p:
        return "retail"
    return "other"


def _dev_from_baseline(sd, building, now) -> float:
    try:
        base = sd.baseline(building, now.hour, now.weekday())
    except KeyError:
        return 0.0
    if not np.isfinite(base) or base <= 0:
        return 0.0
    series = sd.building_series(building)
    idx = series.index.searchsorted(now, side="right") - 1
    if idx < 0:
        return 0.0
    load = float(series.iloc[idx])
    if not np.isfinite(load):
        return 0.0
    return float((load - base) / base)


def _type_map(sd) -> dict:
    meta = sd.metadata
    out = {}
    for _, row in meta.iterrows():
        b = row.get("building_id")
        if b is not None:
            out[str(b)] = building_type_code(row.get("primaryspaceusage", ""))
    return out


def _z_peers(sd, building, now) -> float:
    tmap = _type_map(sd)
    code = tmap.get(building, "other")
    peers = [b for b in sd.buildings if b != building and tmap.get(b, "other") == code]
    if len(peers) < 2:
        return 0.0
    vals = []
    for p in peers:
        s = sd.building_norm_series(p)
        idx = s.index.searchsorted(now, side="right") - 1
        if idx < 0:
            continue
        v = float(s.iloc[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return 0.0
    mean, std = float(np.mean(vals)), float(np.std(vals))
    if std <= 1e-9:
        return 0.0
    s = sd.building_norm_series(building)
    idx = s.index.searchsorted(now, side="right") - 1
    if idx < 0:
        return 0.0
    val = float(s.iloc[idx])
    if not np.isfinite(val):
        return 0.0
    return float((val - mean) / std)


def _norm_load(sd, building, now) -> float:
    s = sd.building_norm_series(building)
    idx = s.index.searchsorted(now, side="right") - 1
    if idx < 0:
        return 0.0
    v = float(s.iloc[idx])
    return v if np.isfinite(v) else 0.0


def _peer_count(sd, building) -> float:
    tmap = _type_map(sd)
    code = tmap.get(building, "other")
    peers = [b for b in sd.buildings if b != building and tmap.get(b, "other") == code]
    return float(len(peers))


PRIORITY = ["beneficial", "harmful", "negligible"]


def build_record(sd, settings: dict, messages: list, building: str, now) -> dict:
    """Build a decision-ready feature record from grouped messages + building context."""
    counts = {k: 0 for k in AGENT_KEYS}
    devs = []
    confs = []
    high = low = 0
    total = len(messages)
    for m in messages:
        counts[m.agent] = counts.get(m.agent, 0) + 1
        devs.append(float(m.deviation_magnitude))
        confs.append(float(m.confidence))
        exp = (m.explanation or "").lower()
        if "high" in exp or "spike" in exp or "increase" in exp:
            high += 1
        if "low" in exp or "decrease" in exp:
            low += 1
    disagreement = 1.0 if (high > 0 and low > 0) else 0.0
    max_dev = max(devs) if devs else 0.0
    mean_dev = float(np.mean(devs)) if devs else 0.0
    max_conf = max(confs) if confs else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0

    meta = sd.metadata.set_index("building_id")
    code = "other"
    log_sqm = 0.0
    if building in meta.index:
        row = meta.loc[building]
        code = building_type_code(row.get("primaryspaceusage", ""))
        sqm = row.get("sqm")
        try:
            log_sqm = float(np.log1p(float(sqm))) if np.isfinite(float(sqm)) else 0.0
        except (TypeError, ValueError):
            log_sqm = 0.0

    f = {
        "n_fixed": float(counts["fixed_limit"]),
        "n_weather": float(counts["weather_adjusted"]),
        "n_peer": float(counts["peer"]),
        "n_trend": float(counts["trend"]),
        "total_agents": float(total),
        "max_deviation": round(max_dev, 4),
        "mean_deviation": round(mean_dev, 4),
        "max_confidence": round(max_conf, 4),
        "mean_confidence": round(mean_conf, 4),
        "high_direction": float(high),
        "low_direction": float(low),
        "disagreement": disagreement,
        "norm_load": round(_norm_load(sd, building, now), 4),
        "dev_baseline": round(_dev_from_baseline(sd, building, now), 4),
        "z_score": round(_z_peers(sd, building, now), 4),
        "temperature": round(float(_temp(sd, now)), 2),
        "weekend": float(1 if now.weekday() >= 5 else 0),
        "hour_of_day": float(now.hour),
        "weekday": float(now.weekday()),
        "log_sqm": round(log_sqm, 3),
        "peer_count": float(_peer_count(sd, building)),
    }
    for t in TYPE_DUMMIES:
        f[f"type_{t}"] = float(1 if code == t else 0)
    return f


def _temp(sd, now) -> float:
    s = sd.weather.get("airTemperature")
    if s is None or s.empty:
        return 0.0
    idx = s.index.searchsorted(now, side="right") - 1
    if idx < 0:
        return 0.0
    v = float(s.iloc[idx])
    return v if np.isfinite(v) else 0.0


FEATURE_COLUMNS = [
    "n_fixed", "n_weather", "n_peer", "n_trend", "total_agents",
    "max_deviation", "mean_deviation", "max_confidence", "mean_confidence",
    "high_direction", "low_direction", "disagreement",
    "norm_load", "dev_baseline", "z_score", "temperature",
    "weekend", "hour_of_day", "weekday", "log_sqm", "peer_count",
] + [f"type_{t}" for t in TYPE_DUMMIES]
