"""Weather-adjusted agent: flag load deviations that temperature alone does not explain."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..messages import AgentMessage
from .base import BaseAgent, linear_fit


class WeatherAdjustedAgent(BaseAgent):
    name = "weather_adjusted"
    message_type = "WEATHER_DEVIATION"

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._calib: dict[str, dict] | None = None

    def _build_calibration(self, sd):
        """Fit norm_load ~ airTemperature per building on a fixed calibration window."""
        cal = {}
        all_ts = sd.meter["timestamp"]
        t0 = all_ts.min()
        t_half = t0 + (all_ts.max() - t0) * 0.6
        temp = sd.weather["airTemperature"]
        df = sd.meter[["timestamp", "building_id", "load_norm"]].copy()
        df["temp"] = df["timestamp"].map(temp).astype(float)
        calib = df[df["timestamp"] < t_half]
        for building, grp in calib.groupby("building_id"):
            x = grp["temp"].values
            y = grp["load_norm"].values
            slope, intercept, r2 = linear_fit(x, y)
            pred = slope * x + intercept
            resid = y - pred
            resid = resid[np.isfinite(resid)]
            cal[building] = {
                "slope": slope, "intercept": intercept, "r2": r2,
                "std": float(resid.std()) if resid.size else 1.0,
            }
        return cal

    def emit(self, sd, now: pd.Timestamp) -> list[AgentMessage]:
        if self._calib is None:
            self._calib = self._build_calibration(sd)
        threshold = float(self.settings.get("weather_std_threshold", 2.0))
        now_temp = self.latest_weather(sd.weather, "airTemperature", now)
        out = []
        for building in sd.buildings:
            cal = self._calib.get(building)
            if not cal or not np.isfinite(now_temp) or cal["std"] <= 0:
                continue
            norm = self.latest_norm(sd, building, now)
            if not np.isfinite(norm):
                continue
            pred = cal["slope"] * now_temp + cal["intercept"]
            resid = norm - pred
            sig = resid / cal["std"]
            if abs(sig) > threshold:
                direction = "high" if sig > 0 else "low"
                out.append(AgentMessage(
                    agent=self.name, timestamp=now, message_type=self.message_type,
                    site=sd.site, building=building,
                    deviation_magnitude=round(float(abs(resid)), 4),
                    confidence=round(float(np.clip(abs(sig) / (threshold * 2), 0.0, 1.0)), 3),
                    explanation=(
                        f"Normalised load is {sig:.1f} std devs {direction} of what temperature "
                        f"(temp {now_temp:.1f} C) predicts for this building."
                    ),
                    input_features={
                        "norm_load": round(norm, 4),
                        "predicted_norm": round(float(pred), 4),
                        "temperature": round(now_temp, 2),
                        "residual": round(float(resid), 4),
                        "residual_std": round(cal["std"], 4),
                    },
                ))
        return out
