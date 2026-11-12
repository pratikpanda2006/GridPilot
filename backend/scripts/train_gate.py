"""Build the replay-labelled training set and train the gate.

  python scripts/train_gate.py [--site Robin] [--step 48] [--max-steps 400]

Prints class distribution, chronological split sizes, confusion matrix and
per-class metrics. Saves the model artifact + metadata.
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app import data
from app.agents import build_agents
from app.config import CANONICAL_CLASSES, DEFAULT_SETTINGS
from app.features import FEATURE_COLUMNS
from app.gate import GateModel
from app.labelling import action_value, label_value
from app.orchestrator import Orchestrator


def build_dataset(site: str, step_h: int, max_steps: int, pos_usd: float | None = None, neg_usd: float | None = None):
    sd = data.load_site(site)
    settings = dict(DEFAULT_SETTINGS)
    settings["site"] = site
    agents = build_agents(settings)
    orch = Orchestrator(settings)

    t0 = sd.meter["timestamp"].min()
    t_max = sd.meter["timestamp"].max()
    # Reserve last 72h to guarantee complete future observation window for counterfactual evaluation
    t_end = t_max - pd.Timedelta(hours=72)
    total_hours = int((t_end - t0).total_seconds() // 3600)
    steps = min(max_steps, total_hours // step_h)
    
    pos_th = float(pos_usd if pos_usd is not None else settings.get("label_positive_usd", 100.0))
    neg_th = float(neg_usd if neg_usd is not None else settings.get("label_negative_usd", -1000.0))

    rows = []
    for i in range(steps):
        now = t0 + pd.Timedelta(hours=i * step_h)
        all_msgs = []
        for a in agents:
            all_msgs.extend(agents[a].emit(sd, now))
        candidates, _ = orch.process(sd, now, all_msgs)
        for cand in candidates:
            val = action_value(sd, cand["building"], pd.Timestamp(cand["timestamp"]),
                               cand["messages"], settings)
            if not np.isfinite(val):
                continue  # missing future observations are excluded from training
            lab = label_value(val, pos_th, neg_th)
            f = dict(cand["features"])
            f["building"] = cand["building"]
            f["timestamp"] = cand["timestamp"]
            f["label"] = lab
            f["action_value"] = val
            rows.append(f)

    cols = FEATURE_COLUMNS + ["building", "timestamp", "label", "action_value"]
    df = pd.DataFrame(rows, columns=cols)
    return df, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT_SETTINGS["site"])
    ap.add_argument("--step", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--pos-usd", type=float, default=None, help="Beneficial USD threshold")
    ap.add_argument("--neg-usd", type=float, default=None, help="Harmful USD threshold")
    args = ap.parse_args()

    df, sd = build_dataset(args.site, args.step, args.max_steps, args.pos_usd, args.neg_usd)
    print(f"dataset rows: {len(df)}")
    if df.empty:
        print("ERROR: empty dataset, cannot train")
        sys.exit(1)

    print("class distribution:")
    print(df["label"].value_counts().to_string())

    # chronological split (no temporal leakage)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f"train {len(train)} / test {len(test)}")

    Xtr, ytr = train[FEATURE_COLUMNS], train["label"]
    Xte, yte = test[FEATURE_COLUMNS], test["label"]

    gate = GateModel()
    gate.fit(Xtr, ytr, note=f"site={args.site}, step={args.step}")

    if gate.status == "trained":
        prob = gate.predict_proba(Xte)
        pred = prob.idxmax(axis=1)
        labels = CANONICAL_CLASSES
        print("\nConfusion matrix (test):")
        print(confusion_matrix(yte, pred, labels=labels))
        print("\nClassification report (test):")
        print(classification_report(yte, pred, labels=labels, digits=3, zero_division=0))
    else:
        print(f"\nGATE STATUS: {gate.status} ({gate.meta.get('reason')}) - fallback used")
        print("Fallback scores are heuristic, NOT trained-model probabilities.")

    print(f"\nSaved model status: {gate.status}, score_type: {gate.score_type}")
    print("Artifacts: backend/artifacts/gate_model.txt + gate_meta.json")


if __name__ == "__main__":
    main()
