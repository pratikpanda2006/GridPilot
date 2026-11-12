"""Fixed-limit agent: current demand vs same-hour-last-week baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..messages import AgentMessage
from .base import BaseAgent


class FixedLimitAgent(BaseAgent):
    name = "fixed_limit"
    message_type = "DEMAND_SPIKE"

    def emit(self, sd, now: pd.Timestamp) -> list[AgentMessage]:
        threshold = float(self.settings.get("detect_threshold", 0.25))
        out = []
        for building in sd.buildings:
            try:
                baseline = sd.baseline(building, now.hour, now.weekday())
            except KeyError:
                continue
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            load = self.latest(sd, building, now)
            if not np.isfinite(load):
                continue
            deviation = (load - baseline) / baseline
            if deviation > threshold:
                recent = self.window(sd, building, now, 12)
                above = int((recent > baseline * (1 + threshold)).sum())
                conf = self._conf_from_consistency(above, int(recent.shape[0]))
                out.append(AgentMessage(
                    agent=self.name, timestamp=now, message_type=self.message_type,
                    site=sd.site, building=building,
                    deviation_magnitude=round(float(deviation), 4),
                    confidence=round(conf, 3),
                    explanation=(
                        f"Load is {deviation*100:.1f}% above the same-hour-last-week "
                        f"baseline (baseline {baseline:.1f} kWh)."
                    ),
                    input_features={
                        "load_kwh": round(float(load), 3),
                        "baseline_kwh": round(float(baseline), 3),
                        "deviation_frac": round(float(deviation), 4),
                        "hour": float(now.hour),
                        "weekday": float(now.weekday()),
                    },
                ))
        return out
