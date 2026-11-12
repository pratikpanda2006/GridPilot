"""FastAPI application – AgentX Smart Message Gate demo.

Endpoints
---------
GET  /api/health                     basic liveness check
GET  /api/info                       site / data info
GET  /api/step                       run one step of the simulation (agents -> gate -> routing)
POST /api/step                       same with explicit timestamp/settings body
POST /api/approve/{candidate_id}     human approval of a VERIFY decision
GET  /api/model/meta                 gate model metadata + importance
GET  /api/replay                     replay a range of historical steps, return outcomes
GET  /api/settings                   get active settings
PATCH /api/settings                  update settings (thresholds etc.)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import build_agents
from .config import DEFAULT_SETTINGS, CANONICAL_CLASSES
from .data import dataset_info, load_site
from .features import build_record
from .gate import gate_model, GateModel, route_decision
from .labelling import action_value, label_value
from .messages import AgentMessage
from .orchestrator import Orchestrator
from .settings_store import settings_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentx.api")

app = FastAPI(title="GridPilot Smart Message Gate", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory pending approvals store  {candidate_id: decision_dict}
# ---------------------------------------------------------------------------
_pending_approvals: dict[str, dict] = {}
_history: list[dict] = []   # store last 200 step results for replay tab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sd():
    site = settings_store.site
    return load_site(site)


def _run_step(now: pd.Timestamp, settings: dict) -> dict:
    sd = load_site(settings.get("site", DEFAULT_SETTINGS["site"]))
    agents = build_agents(settings)
    orch = Orchestrator(settings)

    all_msgs: list[AgentMessage] = []
    agent_counts: dict[str, int] = {}
    for name, agent in agents.items():
        emitted = agent.emit(sd, now)
        all_msgs.extend(emitted)
        agent_counts[name] = len(emitted)

    candidates, stats = orch.process(sd, now, all_msgs)

    decisions = []
    for cand in candidates:
        cid = uuid.uuid4().hex[:10]
        gate_result = gate_model.decide(cand["features"], settings)
        action = gate_result["action"]
        requires_human = gate_result.get("requires_human", False)

        # counterfactual value (may be NaN if future not yet observed)
        try:
            val = action_value(
                sd, cand["building"], pd.Timestamp(cand["timestamp"]),
                cand["messages"], settings,
            )
            pos_th = float(settings.get("label_positive_usd", DEFAULT_SETTINGS["label_positive_usd"]))
            neg_th = float(settings.get("label_negative_usd", DEFAULT_SETTINGS["label_negative_usd"]))
            replay_label = label_value(val, pos_th, neg_th) if np.isfinite(val) else None
            replay_value = round(float(val), 2) if np.isfinite(val) else None
        except Exception:
            replay_label = None
            replay_value = None

        decision = {
            "id": cid,
            "building": cand["building"],
            "timestamp": cand["timestamp"],
            "agents": cand["summary"]["agents"],
            "n_messages": cand["summary"]["n_messages"],
            "max_deviation": cand["summary"]["max_deviation"],
            "mean_confidence": cand["summary"]["mean_confidence"],
            "gate": gate_result,
            "action": action,
            "reason": gate_result["reason"],
            "requires_human": requires_human,
            "messages": cand["messages"],
            "features": {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                         for k, v in cand["features"].items()},
            "replay_label": replay_label,
            "replay_value_usd": replay_value,
            "approved_by_human": None,  # filled in /approve
        }
        if requires_human:
            _pending_approvals[cid] = decision

        decisions.append(decision)

    result = {
        "step_time": now.isoformat(),
        "agent_counts": agent_counts,
        "orchestrator_stats": stats,
        "decisions": decisions,
    }
    _history.append(result)
    if len(_history) > 200:
        _history.pop(0)
    return result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    hours_offset: int | None = None   # hours from dataset start
    timestamp: str | None = None      # ISO timestamp override
    settings_patch: dict | None = None


class ApprovalRequest(BaseModel):
    approved: bool
    operator_note: str = ""


class SettingsPatch(BaseModel):
    patch: dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "model": gate_model.status, "score_type": gate_model.score_type}


@app.get("/api/info")
def info():
    s = settings_store.get()
    return dataset_info(s["site"])


@app.get("/api/settings")
def get_settings():
    return settings_store.get()


@app.patch("/api/settings")
def patch_settings(body: SettingsPatch):
    try:
        return settings_store.update(body.patch)
    except KeyError as e:
        raise HTTPException(400, str(e))


@app.get("/api/model/meta")
def model_meta():
    return {
        "status": gate_model.status,
        "score_type": gate_model.score_type,
        "meta": gate_model.meta,
        "importance": dict(sorted(gate_model.importance.items(), key=lambda kv: kv[1], reverse=True)[:15]),
        "classes": gate_model.classes,
    }


@app.get("/api/step")
def get_step(hours_offset: int = Query(5000, description="Hours from dataset start")):
    s = settings_store.get()
    sd = load_site(s["site"])
    t0 = sd.meter["timestamp"].min()
    now = t0 + pd.Timedelta(hours=hours_offset)
    return _run_step(now, s)


@app.post("/api/step")
def post_step(body: StepRequest):
    s = settings_store.get()
    if body.settings_patch:
        try:
            s.update(body.settings_patch)
        except Exception:
            pass
    sd = load_site(s["site"])
    t0 = sd.meter["timestamp"].min()
    if body.timestamp:
        now = pd.Timestamp(body.timestamp)
    elif body.hours_offset is not None:
        now = t0 + pd.Timedelta(hours=body.hours_offset)
    else:
        now = t0 + pd.Timedelta(hours=5000)
    return _run_step(now, s)


@app.post("/api/approve/{candidate_id}")
def approve(candidate_id: str, body: ApprovalRequest):
    if candidate_id not in _pending_approvals:
        raise HTTPException(404, f"Candidate {candidate_id!r} not in pending approvals")
    _pending_approvals[candidate_id]["approved_by_human"] = body.approved
    _pending_approvals[candidate_id]["operator_note"] = body.operator_note
    _pending_approvals[candidate_id]["requires_human"] = False   # resolved
    return {"status": "recorded", "decision": _pending_approvals[candidate_id]}


@app.get("/api/replay")
def get_replay(
    start_offset: int = Query(1000, description="Start hours offset from dataset start"),
    end_offset: int = Query(10000, description="End hours offset from dataset start"),
    step_h: int = Query(72, description="Step size in hours"),
    max_points: int = Query(60, description="Max data points to return"),
):
    """Run the full pipeline over a historical range and return outcomes for comparison."""
    s = settings_store.get()
    sd = load_site(s["site"])
    t0 = sd.meter["timestamp"].min()
    t_max = sd.meter["timestamp"].max() - pd.Timedelta(hours=72)

    start = t0 + pd.Timedelta(hours=start_offset)
    end = min(t0 + pd.Timedelta(hours=end_offset), t_max)

    agents = build_agents(s)
    orch = Orchestrator(s)

    pts: list[dict] = []
    cur = start
    pos_th = float(s.get("label_positive_usd", DEFAULT_SETTINGS["label_positive_usd"]))
    neg_th = float(s.get("label_negative_usd", DEFAULT_SETTINGS["label_negative_usd"]))
    count = 0
    while cur <= end and count < max_points:
        all_msgs: list[AgentMessage] = []
        for ag in agents.values():
            all_msgs.extend(ag.emit(sd, cur))
        cands, stats = orch.process(sd, cur, all_msgs)
        for cand in cands:
            gate_r = gate_model.decide(cand["features"], s)
            try:
                val = action_value(sd, cand["building"], pd.Timestamp(cand["timestamp"]),
                                   cand["messages"], s)
                replay_label = label_value(val, pos_th, neg_th) if np.isfinite(val) else None
            except Exception:
                val, replay_label = float("nan"), None
            pts.append({
                "timestamp": cand["timestamp"],
                "building": cand["building"],
                "action": gate_r["action"],
                "predicted": gate_r["predicted"],
                "confidence": gate_r["confidence"],
                "replay_label": replay_label,
                "replay_value_usd": round(float(val), 2) if np.isfinite(val) else None,
                "agents": cand["summary"]["agents"],
                "n_messages": cand["summary"]["n_messages"],
            })
        cur += pd.Timedelta(hours=step_h)
        count += 1

    correct = sum(1 for p in pts if p["replay_label"] and p["predicted"] == p["replay_label"])
    total_labelled = sum(1 for p in pts if p["replay_label"])
    return {
        "points": pts,
        "total_decisions": len(pts),
        "labelled": total_labelled,
        "correct_predictions": correct,
        "accuracy": round(correct / total_labelled, 3) if total_labelled else None,
    }


# ---------------------------------------------------------------------------
# Serve the demo SPA from /
# ---------------------------------------------------------------------------

@app.get("/logo.png")
def get_logo():
    logo_path = Path(__file__).resolve().parent.parent / "static" / "logo.png"
    return FileResponse(logo_path)

@app.get("/bg_ref.png")
def get_bg_ref():
    bg_path = Path(__file__).resolve().parent.parent / "static" / "bg_ref.png"
    return FileResponse(bg_path)

@app.get("/", response_class=HTMLResponse)
def demo_ui():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")
