"""Application-wide constants and paths (AgentX Smart Message Gate)."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Data root. Falls back so the app works whether sites are extracted or zipped.
SITES_ROOT = PROJECT_ROOT / "bdg2" / "sites_extracted" / "sites"
SITES_ZIP = PROJECT_ROOT / "sites.zip"

ARTIFACT_DIR = BACKEND_DIR / "artifacts"
CACHE_DIR = BACKEND_DIR / "cache"
MODEL_PATH = ARTIFACT_DIR / "gate_model.txt"
MODEL_META_PATH = ARTIFACT_DIR / "gate_meta.json"
LABELS_CACHE = CACHE_DIR / "labels.parquet"

DEFAULT_SITE = "Robin"
SITE_ALIASES = {"bdg2": "Robin"}

# --- Canonical class definitions ---
CANONICAL_CLASSES = ["beneficial", "harmful", "negligible"]
CLASS_TO_IDX = {"beneficial": 0, "harmful": 1, "negligible": 2}
IDX_TO_CLASS = {0: "beneficial", 1: "harmful", 2: "negligible"}

# --- data-integrity defaults (from the methodology) ---
BUILDING_VALID_FRACTION = 0.80
SHORT_GAP_INTERP_HOURS = 3          # only gaps up to this many hours are interpolated
SPARSE_WEATHER_FEATURES = ["cloudCoverage", "precipDepth6HR"]
HOURS_PERSIST_START = 24            # counterfactual window start (hours after message)
HOURS_PERSIST_END = 72              # counterfactual window end

# --- default economic assumptions ---
DEFAULT_SETTINGS = {
    "site": "Robin",
    "detect_threshold": 0.25,        # fixed-limit agent threshold (25% above baseline)
    "weather_std_threshold": 2.0,    # weather-adjusted residual threshold (std devs)
    "peer_z_threshold": 2.0,         # peer agent z-score threshold
    "trend_window_hours": 288,       # trend look-back window (12 days)
    "trend_min_slope_per_day": 0.008,  # min daily trend slope (fraction/day)
    "trend_min_r2": 0.40,            # min fit quality
    "trend_min_consistency": 0.60,   # min day-to-day directional consistency
    "accept_prob": 0.60,             # P(beneficial) needed to accept
    "verify_max_prob": 0.72,         # below this top probability -> VERIFY
    "ignore_harm_prob": 0.45,        # P(harmful) above this -> IGNORE
    "discount_neg_prob": 0.55,       # P(negligible) above this -> DISCOUNT
    "max_hours_age": 12,             # stale after 12h
    "label_positive_usd": 100.0,     # >= this action value -> beneficial
    "label_negative_usd": -1000.0,   # <= this action value -> harmful (commercial building scale)
    "shortfall_multiplier": 1.5,
    "resale_fraction": 0.5,
    "tariff_per_kwh": 0.12,          # $/kWh
    "visit_cost": 150.0,             # $ per field visit
    "replay_speed": 8,               # steps per tick (hours)
    "inject_faults": False,
}
