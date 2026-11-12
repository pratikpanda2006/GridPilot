"""Structured message model used by all helper agents, the orchestrator and the gate."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

AGENT_NAMES = ("fixed_limit", "weather_adjusted", "peer", "trend")
MESSAGE_TYPES = ("DEMAND_SPIKE", "WEATHER_DEVIATION", "PEER_OUTLIER", "PERSISTENT_TREND")


class AgentMessage(BaseModel):
    """A single structured message emitted by a helper agent."""

    agent: str = Field(description="source agent name")
    timestamp: datetime = Field(description="decision/message time")
    message_type: str
    site: str
    building: str
    deviation_magnitude: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    message_age: float = Field(default=0.0, ge=0, description="hours old")
    explanation: str = ""
    input_features: dict[str, float] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    injected: bool = Field(default=False, description="true if this fault was deliberately injected")

    @field_validator("agent")
    @classmethod
    def _agent_known(cls, v):
        if v not in AGENT_NAMES:
            raise ValueError(f"unknown agent '{v}'")
        return v

    @field_validator("message_type")
    @classmethod
    def _type_known(cls, v):
        if v not in MESSAGE_TYPES:
            raise ValueError(f"unknown message type '{v}'")
        return v

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["timestamp"] = self.timestamp.isoformat()
        return d
