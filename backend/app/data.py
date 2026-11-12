"""Data loading and integrity preparation for BDG2 site files.

Loads individual site meter / weather / metadata CSVs, applies the
cleaning methodology from the deck:
  - keep buildings with >80% valid readings
  - interpolate short gaps only, leave long gaps missing
  - normalise each building's load by its own mean
  - drop intentionally sparse weather features (cloudCoverage, precipDepth6HR)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("agentx.data")


def list_sites() -> list[str]:
    """Return available BDG2 site names."""
    root = config.SITES_ROOT
    if root.is_dir():
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
        if names:
            return names
    # fallback: names inside sites.zip
    import zipfile
    with zipfile.ZipFile(config.SITES_ZIP) as z:
        names = sorted({
            Path(n).parts[1] for n in z.namelist()
            if n.count("/") == 2
        })
    return names


@dataclass
class SiteData:
    site: str
    meter: pd.DataFrame            # long frame: timestamp, building_id, load, load_norm
    weather: pd.DataFrame          # index=timestamp, cleaned columns
    metadata: pd.DataFrame         # one row per building
    buildings: list[str]
    prep: dict = field(default_factory=dict)
    _baseline: pd.DataFrame | None = None
    _s_cache: dict = field(default_factory=dict)
    _sn_cache: dict = field(default_factory=dict)

    # ---- convenience ----
    def building_series(self, building: str) -> pd.Series:
        if building not in self._s_cache:
            s = self.meter[self.meter["building_id"] == building].set_index("timestamp")["load"]
            self._s_cache[building] = s
        return self._s_cache[building]

    def building_norm_series(self, building: str) -> pd.Series:
        if building not in self._sn_cache:
            s = self.meter[self.meter["building_id"] == building].set_index("timestamp")["load_norm"]
            self._sn_cache[building] = s
        return self._sn_cache[building]

    def building_meta(self, building: str) -> dict:
        row = self.metadata[self.metadata["building_id"] == building]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def baseline(self, building: str, hour: int, weekday: int) -> float:
        """Same-hour-last-week baseline (median of that weekday/hour over past weeks)."""
        return float(self._baseline[(building, weekday, hour)])


def _load_site_raw(site: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = config.SITES_ROOT / site
    meter = pd.read_csv(root / "meter.csv")
    weather = pd.read_csv(root / "weather.csv")
    metadata = pd.read_csv(root / "metadata.csv")
    return meter, weather, metadata


def _melt_meter(meter_wide: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(meter_wide["timestamp"], errors="coerce")
    df = meter_wide.drop(columns=["timestamp"])
    melted = df.melt(var_name="building_id", value_name="load", ignore_index=False)
    melted["timestamp"] = np.repeat(ts.values, len(df.columns))
    melted["load"] = pd.to_numeric(melted["load"], errors="coerce")
    return melted[["timestamp", "building_id", "load"]].reset_index(drop=True)


def _clean_meter(meter_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    meter_long = meter_long.dropna(subset=["timestamp"])
    # per-building validity
    total = meter_long.groupby("building_id")["load"].size()
    valid_ = meter_long.groupby("building_id")["load"].apply(lambda s: int(s.notna().sum()))
    frac = valid_ / total
    keep = frac[frac >= config.BUILDING_VALID_FRACTION].index.tolist()
    keep = [b for b in keep if b in frac.index]
    m = meter_long[meter_long["building_id"].isin(keep)].copy()
    m = m.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    # interpolate short gaps only (per building, time-ordered)
    m["load"] = (
        m.groupby("building_id")["load"]
        .transform(lambda s: s.interpolate(method="linear", limit=config.SHORT_GAP_INTERP_HOURS, limit_direction="both"))
    )
    # mean-load normalisation (per building, valid readings)
    bmean = m.groupby("building_id")["load"].transform("mean")
    m["load_norm"] = m["load"] / bmean
    prep = {
        "buildings_raw": int(total.shape[0]),
        "buildings_kept": len(keep),
        "buildings_dropped": int(total.shape[0] - len(keep)),
        "valid_readings": int(m["load"].notna().sum()),
        "total_rows": int(m.shape[0]),
    }
    return m, prep


def _clean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
    w = w.dropna(subset=["timestamp"])
    w = w.drop(columns=[c for c in config.SPARSE_WEATHER_FEATURES if c in w.columns])
    w = w.set_index("timestamp").sort_index()
    # coerce numeric columns; drop genuinely non-numeric leftovers
    for col in w.columns:
        converted = pd.to_numeric(w[col], errors="coerce")
        if converted.notna().mean() >= 0.5:
            w[col] = converted
        else:
            w = w.drop(columns=[col])
    w = w.apply(lambda s: s.interpolate(method="linear", limit=6, limit_direction="both"))
    return w


def _compute_baseline(meter: pd.DataFrame) -> pd.DataFrame:
    """Median load per (building, weekday, hour) — the same-hour-last-week level."""
    m = meter.copy()
    m["weekday"] = m["timestamp"].dt.weekday
    m["hour"] = m["timestamp"].dt.hour
    base = m.groupby(["building_id", "weekday", "hour"])["load"].median()
    return base


_CACHE: dict[str, SiteData] = {}
_CACHE_META: dict[str, dict] = {}


def load_site(site: str, use_cache: bool = True) -> SiteData:
    site = config.SITE_ALIASES.get(site, site)
    if use_cache and site in _CACHE:
        return _CACHE[site]
    meter_w, weather_w, meta = _load_site_raw(site)
    meter_long = _melt_meter(meter_w)
    meter_clean, prep = _clean_meter(meter_long)
    weather_clean = _clean_weather(weather_w)
    baseline = _compute_baseline(meter_clean)
    buildings = list(meter_clean["building_id"].drop_duplicates())
    sd = SiteData(
        site=site,
        meter=meter_clean,
        weather=weather_clean,
        metadata=meta,
        buildings=buildings,
        prep=prep,
    )
    sd._baseline = baseline
    if use_cache:
        _CACHE[site] = sd
    return sd


def dataset_info(site: str) -> dict:
    sd = load_site(site)
    return {
        "site": sd.site,
        "buildings": len(sd.buildings),
        "buildings_kept": sd.prep["buildings_kept"],
        "buildings_dropped": sd.prep["buildings_dropped"],
        "available_hours": int(sd.meter["timestamp"].nunique()),
        "time_start": str(sd.meter["timestamp"].min()),
        "time_end": str(sd.meter["timestamp"].max()),
        "valid_readings": sd.prep["valid_readings"],
        "weather_features": list(sd.weather.columns),
        "metadata_columns": list(sd.metadata.columns),
    }
