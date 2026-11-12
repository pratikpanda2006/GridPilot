"""Counterfactual replay labelling.

For each message we compare the cost of ACTING on it against the cost of NOT
ACTING, both evaluated against the REALISED future load (next 24-72 h).

Plans (energy to procure for the window):
  baseline_plan = energy procured without acting = baseline energy (same-hour
                  last-week profile summed over the window)
  action_plan   = energy procured after acting = baseline_plan * (1 + uplift)

Cost of a plan P when realised load is A:
  cost(P, A) = tariff * P
               - resale_fraction * tariff * max(0, P - A)   # surplus resale
               + shortfall_multiplier * tariff * max(0, A - P) # shortfall penalty

  action_value = cost(baseline_plan, A) - cost(action_plan, A)

Positive action_value => acting reduced cost (BENEFICIAL).
Negative action_value => acting increased cost (HARMFUL).
Near zero            => NEGLIGIBLE.

Only actual measurements are used for realised load. Missing future windows are
excluded from training (never labelled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CANONICAL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

# Canonical class mapping: beneficial=0, harmful=1, negligible=2
LABELS = CLASS_TO_IDX
LABEL_INDEX = IDX_TO_CLASS


def plan_energy(sd, building: str, t, settings: dict) -> float:
    s = sd.building_series(building)
    w_start = t + pd.Timedelta(hours=24)
    w_end = t + pd.Timedelta(hours=72)
    idx = s[(s.index >= w_start) & (s.index <= w_end)].index
    if len(idx) == 0:
        return float("nan")
    try:
        return float(np.nansum([sd.baseline(building, ts.hour, ts.weekday()) for ts in idx]))
    except KeyError:
        return float("nan")


def realised_energy(sd, building: str, t) -> float:
    s = sd.building_series(building)
    w_start = t + pd.Timedelta(hours=24)
    w_end = t + pd.Timedelta(hours=72)
    actual = s[(s.index >= w_start) & (s.index <= w_end)]
    if actual.empty:
        return float("nan")
    return float(actual.sum())


def plan_cost(P: float, A: float, tariff: float, rf: float, sf: float) -> float:
    if np.isnan(P) or np.isnan(A):
        return float("nan")
    surplus = max(0.0, P - A)
    shortfall = max(0.0, A - P)
    return tariff * P - rf * tariff * surplus + sf * tariff * shortfall


def signed_uplift(messages) -> float:
    """Signed procurement uplift implied by a message group (direction-aware)."""
    total = 0.0
    for m in messages:
        if isinstance(m, dict):
            d = float(m.get("deviation_magnitude", 0.0))
            exp = (m.get("explanation") or "").lower()
        else:
            d = float(m.deviation_magnitude)
            exp = (m.explanation or "").lower()
        if any(w in exp for w in ("low", "decrease")):
            total -= d
        else:
            total += d
    return float(np.clip(total, -0.8, 3.0))


def action_value(sd, building: str, t, messages, settings: dict) -> float:
    base = plan_energy(sd, building, t, settings)
    actual = realised_energy(sd, building, t)
    if np.isnan(base) or np.isnan(actual):
        return float("nan")
    tariff = float(settings.get("tariff_per_kwh", 0.12))
    rf = float(settings.get("resale_fraction", 0.5))
    sf = float(settings.get("shortfall_multiplier", 1.5))
    uplift = signed_uplift(messages)
    action_plan = base * (1.0 + uplift)
    baseline_cost = plan_cost(base, actual, tariff, rf, sf)
    action_cost = plan_cost(action_plan, actual, tariff, rf, sf)
    return float(baseline_cost - action_cost)


def label_value(value: float, pos_usd: float, neg_usd: float) -> str:
    if not np.isfinite(value):
        return "negligible"
    if value >= pos_usd:
        return "beneficial"
    if value <= neg_usd:
        return "harmful"
    return "negligible"
