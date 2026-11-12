"""Thread-safe runtime settings store. All settings here actually affect decisions."""
from __future__ import annotations

import threading

from .config import DEFAULT_SETTINGS
from .data import list_sites

_VALID_KEYS = set(DEFAULT_SETTINGS.keys())

class SettingsStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = dict(DEFAULT_SETTINGS)

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def update(self, patch: dict) -> dict:
        with self._lock:
            for k, v in patch.items():
                if k not in _VALID_KEYS:
                    raise KeyError(f"unknown setting '{k}'")
                self._data[k] = v
            return dict(self._data)

    @property
    def site(self) -> str:
        with self._lock:
            return self._data["site"]

    def set_site(self, site: str):
        sites = list_sites()
        if site not in sites:
            raise ValueError(f"unknown site '{site}'")
        with self._lock:
            self._data["site"] = site


settings_store = SettingsStore()
