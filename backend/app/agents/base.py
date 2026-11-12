"""Common scaffolding for helper agents."""
from __future__ import annotations

import numpy as np
import pandas as pd


class BaseAgent:
    name: str = "base"
    message_type: str = ""

    def __init__(self, settings: dict):
        self.settings = settings

    def emit(self, sd, now: pd.Timestamp) -> list:
        raise NotImplementedError

    # ---- helpers ----
    @staticmethod
    def window(sd, building: str, now: pd.Timestamp, hours: float) -> pd.Series:
        """Raw load for building in the window ending at `now` (no future data)."""
        m = sd.building_series(building)
        return m[(m.index <= now) & (m.index >= now - pd.Timedelta(hours=hours))]

    @staticmethod
    def window_norm(sd, building: str, now: pd.Timestamp, hours: float) -> pd.Series:
        m = sd.building_norm_series(building)
        return m[(m.index <= now) & (m.index >= now - pd.Timedelta(hours=hours))]

    @staticmethod
    def latest(sd, building: str, now: pd.Timestamp) -> float:
        """Most recent raw load at or before `now` (handles missing hours)."""
        s = sd.building_series(building)
        idx = s.index.searchsorted(now, side="right") - 1
        if idx < 0:
            return float("nan")
        return float(s.iloc[idx])

    @staticmethod
    def latest_norm(sd, building: str, now: pd.Timestamp) -> float:
        s = sd.building_norm_series(building)
        idx = s.index.searchsorted(now, side="right") - 1
        if idx < 0:
            return float("nan")
        return float(s.iloc[idx])

    @staticmethod
    def latest_weather(weather: pd.DataFrame, col: str, now: pd.Timestamp) -> float:
        if col not in weather.columns or weather.empty:
            return float("nan")
        s = weather[col]
        idx = s.index.searchsorted(now, side="right") - 1
        if idx < 0:
            return float("nan")
        return float(s.iloc[idx])

    def _conf_from_consistency(self, above: float, n: int) -> float:
        if n == 0:
            return 0.5
        return float(np.clip(0.5 + 0.5 * (above / n), 0.0, 1.0))


def linear_fit(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r2) for a simple least-squares fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(np.clip(r2, 0.0, 1.0))
