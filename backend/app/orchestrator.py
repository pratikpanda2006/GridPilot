"""Message orchestration: validate, group, enrich, and build decision records."""
from __future__ import annotations

import pandas as pd

from .features import build_record
from .messages import AgentMessage


class Orchestrator:
    def __init__(self, settings: dict):
        self.settings = settings

    def process(self, sd, now, messages: list[AgentMessage]) -> tuple[list[dict], dict]:
        valid = []
        invalid = 0
        stale = 0
        max_age = float(self.settings.get("max_hours_age", 12))
        known_buildings = set(sd.buildings)
        for m in messages:
            if not isinstance(m, AgentMessage):
                invalid += 1
                continue
            if m.site != sd.site:
                invalid += 1
                continue
            if m.building not in known_buildings:
                invalid += 1
                continue
            # drop stale messages
            age_hours = (now - pd.Timestamp(m.timestamp)).total_seconds() / 3600.0
            if age_hours > max_age:
                stale += 1
                continue
            valid.append(m)

        # group by (building, hour bucket)
        groups: dict[tuple, list[AgentMessage]] = {}
        for m in valid:
            bucket = pd.Timestamp(m.timestamp).floor("h")
            groups.setdefault((m.building, bucket), []).append(m)

        candidates = []
        for (building, bucket), msgs in groups.items():
            feats = build_record(sd, self.settings, msgs, building, bucket)
            candidates.append({
                "building": building,
                "timestamp": bucket.isoformat(),
                "features": feats,
                "messages": [m.to_dict() for m in msgs],
                "summary": {
                    "n_messages": len(msgs),
                    "agents": sorted({m.agent for m in msgs}),
                    "max_deviation": feats["max_deviation"],
                    "mean_confidence": feats["mean_confidence"],
                },
            })

        stats = {
            "received": len(messages),
            "valid": len(valid),
            "invalid": invalid,
            "stale": stale,
            "candidates": len(candidates),
        }
        return candidates, stats
