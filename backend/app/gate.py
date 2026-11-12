"""Intelligence gate: three-class LightGBM classifier with an explicit fallback.

The fallback is a deterministic heuristic classifier. Its output is *never*
presented as trained-model probabilities: every prediction carries a status
("trained" vs "fallback") and a score_type field.
"""
from __future__ import annotations

import json
import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import (
    ARTIFACT_DIR,
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    DEFAULT_SETTINGS,
    IDX_TO_CLASS,
    MODEL_META_PATH,
    MODEL_PATH,
)
from .features import FEATURE_COLUMNS, TYPE_DUMMIES

logger = logging.getLogger("agentx.gate")

CLASSES = CANONICAL_CLASSES
MIN_CLASS_SAMPLES = 5


def route_decision(probs: dict[str, float], settings: dict, features: dict | None = None) -> dict:
    """Four-way routing based on gate probabilities and discom operational thresholds.

    Actions:
      - ACCEPT: P(beneficial) >= accept_prob and top probability >= verify_max_prob.
      - IGNORE: P(harmful) >= ignore_harm_prob (asymmetric penalty protection).
      - DISCOUNT: P(negligible) >= discount_neg_prob (alert is routine noise).
      - VERIFY: Top probability < verify_max_prob, or high uncertainty / disagreement.
    """
    p_b = float(probs.get("beneficial", 0.0))
    p_h = float(probs.get("harmful", 0.0))
    p_n = float(probs.get("negligible", 0.0))

    accept_th = float(settings.get("accept_prob", 0.60))
    ignore_th = float(settings.get("ignore_harm_prob", 0.45))
    discount_th = float(settings.get("discount_neg_prob", 0.55))
    verify_max = float(settings.get("verify_max_prob", 0.72))

    # Priority 1: High harmful risk -> IGNORE to avoid costly procurement penalty / wasted visit
    if p_h >= ignore_th:
        action = "IGNORE"
        reason = (
            f"High harmful probability P(harmful)={p_h:.3f} >= {ignore_th:.2f}. "
            "Acting would likely induce surplus resale losses or unnecessary dispatch."
        )
        requires_human = False
    # Priority 2: High beneficial confidence -> ACCEPT
    elif p_b >= accept_th and max(p_b, p_h, p_n) >= verify_max:
        action = "ACCEPT"
        reason = (
            f"High beneficial confidence P(beneficial)={p_b:.3f} >= {accept_th:.2f} "
            f"(top prob {max(p_b, p_h, p_n):.3f} >= {verify_max:.2f}). Safe to act immediately."
        )
        requires_human = False
    # Priority 3: Negligible value -> DISCOUNT
    elif p_n >= discount_th:
        action = "DISCOUNT"
        reason = (
            f"Dominant negligible probability P(negligible)={p_n:.3f} >= {discount_th:.2f}. "
            "Down-weighting signal; no immediate costly intervention justified."
        )
        requires_human = False
    # Priority 4: Uncertainty / borderline / disagreement -> VERIFY (human in the loop)
    else:
        action = "VERIFY"
        top_p = max(p_b, p_h, p_n)
        reason = (
            f"Uncertain or borderline confidence (top P={top_p:.3f} < {verify_max:.2f}). "
            "Requires human operator verification before scheduling field inspection or adjusting trade."
        )
        requires_human = True

    return {
        "action": action,
        "reason": reason,
        "requires_human": requires_human,
        "thresholds_used": {
            "accept_prob": accept_th,
            "ignore_harm_prob": ignore_th,
            "discount_neg_prob": discount_th,
            "verify_max_prob": verify_max,
        },
    }


class GateModel:
    def __init__(self):
        self.model = None
        self.classes = CLASSES
        self.feature_cols = FEATURE_COLUMNS
        self.status = "untrained"
        self.score_type = None
        self.meta = {}
        self.importance = {}

    # ---------------- training ----------------
    def fit(self, X: pd.DataFrame, y: pd.Series, note: str = ""):
        X = X[self.feature_cols].astype(float)
        counts = y.value_counts()
        ok = all(counts.get(c, 0) >= MIN_CLASS_SAMPLES for c in CLASSES)
        if not ok:
            logger.warning("Class imbalance too large to train reliably -> fallback. counts=%s", counts.to_dict())
            self.status = "fallback"
            self.score_type = "heuristic"
            self.meta = {"reason": "insufficient class samples", "class_counts": counts.to_dict(), "note": note}
            return self
        # map labels to class indices using canonical CLASS_TO_IDX
        y_i = y.map(CLASS_TO_IDX).astype(int).values
        try:
            model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=150,
                learning_rate=0.06, num_leaves=15, min_child_samples=5,
                subsample=0.9, colsample_bytree=0.8, random_state=7, verbose=-1,
            )
            model.fit(X.values, y_i, eval_metric="multi_logloss")
        except Exception as e:  # noqa: BLE001
            logger.warning("LightGBM training failed: %s -> fallback", e)
            self.status = "fallback"
            self.score_type = "heuristic"
            self.meta = {"reason": f"LightGBM failed: {e}", "note": note}
            return self
        self.model = model
        self.status = "trained"
        self.score_type = "lightgbm"
        try:
            imp = model.feature_importances_
            self.importance = dict(zip(self.feature_cols, [int(i) for i in imp]))
        except Exception:  # noqa: BLE001
            self.importance = {}
        self.meta = {
            "status": "trained",
            "classes": CLASSES,
            "class_counts": counts.to_dict(),
            "n_train": int(len(X)),
            "note": note,
            "algorithm": "lightgbm-multiclass",
        }
        self.save()
        return self

    def save(self):
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        if self.model is not None:
            try:
                self.model.booster_.save_model(str(MODEL_PATH))
            except Exception:  # noqa: BLE001
                pass
        with open(MODEL_META_PATH, "w") as fh:
            json.dump({"status": self.status, "score_type": self.score_type, "meta": self.meta,
                       "importance": self.importance}, fh, indent=2)

    def load(self):
        if MODEL_PATH.exists():
            try:
                booster = lgb.Booster(model_file=str(MODEL_PATH))
                self.model = lgb.LGBMClassifier()
                self.model._Booster = booster  # noqa: SLF001
                self.status = "trained"
                self.score_type = "lightgbm"
            except Exception as e:  # noqa: BLE001
                logger.warning("could not load model artifact: %s", e)
        if MODEL_META_PATH.exists():
            try:
                with open(MODEL_META_PATH) as fh:
                    meta = json.load(fh)
                self.meta = meta.get("meta", {})
                self.importance = meta.get("importance", {})  # restore importance from saved JSON
                if meta.get("status") == "fallback":
                    self.status = "fallback"
                    self.score_type = "heuristic"
            except Exception:  # noqa: BLE001
                pass
        return self

    # ---------------- prediction ----------------
    def _heuristic_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(X), 3))
        for i, row in X.iterrows():
            mag = float(row.get("mean_deviation", 0.0))
            conf = float(row.get("mean_confidence", 0.0))
            high = float(row.get("high_direction", 0.0))
            low = float(row.get("low_direction", 0.0))
            disag = float(row.get("disagreement", 0.0))
            net = high - low
            b = 0.05 + 0.35 * float(mag > 0.5) + 0.2 * conf + 0.1 * float(net > 0) - 0.12 * disag
            h = 0.05 + 0.25 * float(net < 0) + 0.1 * float(disag > 0)
            n = 0.3 + 0.2 * float(mag < 0.3) + 0.1 * (1 - conf) + 0.08 * disag
            total = b + h + n
            # Order strictly matches CLASSES: ["beneficial", "harmful", "negligible"]
            probs[i] = [b / total, h / total, n / total]
        return probs

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.feature_cols, fill_value=0.0).astype(float)
        if self.status == "trained" and self.model is not None:
            try:
                raw_prob = self.model.predict_proba(X.values)
                # Check model.classes_ when aligning probability columns
                prob_df = pd.DataFrame(0.0, index=X.index, columns=self.classes)
                for col_idx, cls_label in enumerate(self.model.classes_):
                    cls_name = IDX_TO_CLASS.get(cls_label, cls_label) if isinstance(cls_label, (int, np.integer)) else cls_label
                    if cls_name in prob_df.columns:
                        prob_df[cls_name] = raw_prob[:, col_idx]
                row_sums = prob_df.sum(axis=1).replace(0, 1.0)
                prob_df = prob_df.div(row_sums, axis=0)
                return prob_df
            except Exception as e:  # noqa: BLE001
                logger.warning("prediction failed (%s) -> heuristic fallback for this batch", e)
        probs = self._heuristic_proba(X)
        return pd.DataFrame(probs, columns=self.classes, index=X.index)

    def decide(self, features: dict, settings: dict | None = None) -> dict:
        X = pd.DataFrame([features]).reindex(columns=self.feature_cols, fill_value=0.0).astype(float)
        prob = self.predict_proba(X).iloc[0]
        prob_dict = {c: round(float(prob[c]), 4) for c in self.classes}
        predicted = str(prob.idxmax())
        confidence = float(prob.max())
        st = settings or DEFAULT_SETTINGS
        routing = route_decision(prob_dict, st, features)
        out = {
            "status": self.status,
            "score_type": self.score_type,
            "probabilities": prob_dict,
            "predicted": predicted,
            "confidence": round(confidence, 4),
            "action": routing["action"],
            "reason": routing["reason"],
            "requires_human": routing["requires_human"],
            "routing": routing,
            "top_features": self._top_features(features),
        }
        return out

    def _top_features(self, features: dict, k: int = 5) -> list[dict]:
        if not self.importance:
            return []
        im = sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True)[:k]
        res = []
        for name, _ in im:
            res.append({"feature": name, "value": round(float(features.get(name, 0.0)), 4),
                        "importance": int(self.importance[name])})
        return res


gate_model = GateModel().load()
