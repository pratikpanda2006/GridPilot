"""Peer agent: compare a building against similar buildings (type/size) at the same site."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..messages import AgentMessage
from .base import BaseAgent


class PeerAgent(BaseAgent):
    name = "peer"
    message_type = "PEER_OUTLIER"

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._groups: dict[str, list[str]] | None = None

    def _build_groups(self, sd):
        meta = sd.metadata
        groups: dict[str, list[str]] = {}
        if "primaryspaceusage" in meta.columns and meta["primaryspaceusage"].notna().any():
            for kind, grp in meta.groupby("primaryspaceusage"):
                groups[str(kind)] = grp["building_id"].tolist()
            return groups
        # fallback: group by size band (sqm)
        sizes = meta.set_index("building_id")["sqm"].astype(float)
        for b in sd.buildings:
            s = sizes.get(b, np.nan)
            if not np.isfinite(s):
                continue
            band = int(np.log1p(s) // 1)
            groups.setdefault(f"size_band_{band}", []).append(b)
        return groups

    def emit(self, sd, now: pd.Timestamp) -> list[AgentMessage]:
        if self._groups is None:
            self._groups = self._build_groups(sd)
        z_threshold = float(self.settings.get("peer_z_threshold", 2.0))
        out = []
        # Precompute peers' normalised load at this hour per group.
        group_norm = {}
        for kind, members in self._groups.items():
            if len(members) < 2:
                continue
            vals = []
            for b in members:
                v = self.latest_norm(sd, b, now)
                if np.isfinite(v):
                    vals.append(v)
            if len(vals) >= 2:
                group_norm[kind] = np.array(vals)
        for building in sd.buildings:
            kind = None
            for k, members in self._groups.items():
                if building in members:
                    kind = k
                    break
            if kind is None or kind not in group_norm:
                continue
            val = self.latest_norm(sd, building, now)
            if not np.isfinite(val):
                continue
            peers = group_norm[kind]
            mean = float(peers.mean())
            std = float(peers.std())
            if std <= 1e-9:
                continue
            z = (val - mean) / std
            if abs(z) > z_threshold:
                direction = "high" if z > 0 else "low"
                out.append(AgentMessage(
                    agent=self.name, timestamp=now, message_type=self.message_type,
                    site=sd.site, building=building,
                    deviation_magnitude=round(float(abs(z)), 3),
                    confidence=round(float(np.clip(abs(z) / (z_threshold * 2), 0.0, 1.0)), 3),
                    explanation=(
                        f"Consumption is {z:.2f} std devs {direction} of {len(peers)} similar "
                        f"buildings ({kind})."
                    ),
                    input_features={
                        "z_score": round(float(z), 3),
                        "peer_mean": round(mean, 4),
                        "peer_std": round(std, 4),
                        "n_peers": float(len(peers)),
                    },
                ))
        return out
