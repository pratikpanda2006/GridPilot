"""Helper agent registry."""
from __future__ import annotations

from .fixed_limit import FixedLimitAgent
from .peer import PeerAgent
from .trend import TrendAgent
from .weather_adjusted import WeatherAdjustedAgent

AGENTS = {
    "fixed_limit": FixedLimitAgent,
    "weather_adjusted": WeatherAdjustedAgent,
    "peer": PeerAgent,
    "trend": TrendAgent,
}


def build_agents(settings: dict) -> dict[str, object]:
    return {name: cls(settings) for name, cls in AGENTS.items()}
