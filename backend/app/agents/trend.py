"""Trend agent: detect persistent increases/decreases over a rolling window.

Trends are evaluated on *daily-averaged* load so that a single noisy hour
cannot drive the signal, matching the "avoid flagging a single noisy reading"
requirement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..messages import AgentMessage
from .base import BaseAgent, linear_fit


class TrendAgent(BaseAgent):
    name = "trend"
    message_type = "PERSISTENT_TREND"

    def emit(self, sd, now: pd.Timestamp) -> list[AgentMessage]:
        window_h = int(self.settings.get("trend_window_hours", 288))
        min_daily = float(self.settings.get("trend_min_slope_per_day", 0.008))
        min_r2 = float(self.settings.get("trend_min_r2", 0.40))
        min_consistent = float(self.settings.get("trend_min_consistency", 0.60))
        out = []
        for building in sd.buildings:
            s = sd.building_norm_series(building)
            sub = s[(s.index <= now) & (s.index >= now - pd.Timedelta(hours=window_h))].dropna()
            if sub.shape[0] < 120:
                continue
            daily = sub.resample("24h").mean().dropna()
            if daily.shape[0] < 5:
                continue
            x = np.arange(daily.shape[0])
            y = daily.values
            slope, intercept, r2 = linear_fit(x, y)
            diffs = np.diff(y)
            sign_frac = float(np.mean(np.sign(diffs) == np.sign(slope))) if slope != 0 else 0.0
            mean_level = float(np.nanmean(y))
            if mean_level <= 0:
                continue
            norm_daily = abs(slope) / mean_level
            if norm_daily >= min_daily and r2 >= min_r2 and sign_frac >= min_consistent:
                total_change = (slope * (daily.shape[0] - 1)) / mean_level
                direction = "increase" if slope > 0 else "decrease"
                out.append(AgentMessage(
                    agent=self.name, timestamp=now, message_type=self.message_type,
                    site=sd.site, building=building,
                    deviation_magnitude=round(float(abs(total_change)), 3),
                    confidence=round(float(np.clip(r2, 0.0, 1.0)), 3),
                    explanation=(
                        f"Sustained {direction} of {abs(total_change)*100:.1f}% over the last "
                        f"{daily.shape[0]} days of daily-averaged load "
                        f"(R²={r2:.2f}, {sign_frac*100:.0f}% consistent)."
                    ),
                    input_features={
                        "daily_slope_frac": round(float(norm_daily), 5),
                        "r_squared": round(float(r2), 4),
                        "sign_consistency": round(sign_frac, 3),
                        "days": float(daily.shape[0]),
                    },
                ))
        return out
