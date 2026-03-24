from __future__ import annotations

from enum import Enum
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field


class CallType(str, Enum):
    WELLNESS_CHECK = "wellness_check"
    EMERGENCY = "emergency"
    GENERAL_INQUIRY = "general_inquiry"


class WellnessDimension(str, Enum):
    PHYSICAL = "physical"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    NUTRITION = "nutrition"
    SOCIAL = "social"


class ConversationPhase(str, Enum):
    GREETING = "greeting"
    IDENTITY = "identity"
    PHYSICAL = "physical"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    NUTRITION = "nutrition"
    SOCIAL = "social"
    CLOSING = "closing"


class ConcernSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Patient(BaseModel):
    id: str
    first_name: str
    last_name: str
    age: int
    medications: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    emergency_contacts: list[EmergencyContact] = Field(default_factory=list)
    care_notes: str = ""


class EmergencyContact(BaseModel):
    name: str
    relationship: str
    phone: str


Patient.model_rebuild()


class WellnessScore(BaseModel):
    dimension: WellnessDimension
    score: int = Field(ge=1, le=10)
    reasoning: str


class Concern(BaseModel):
    category: str
    severity: ConcernSeverity
    description: str
    suggested_action: str


class ConversationState(BaseModel):
    call_sid: str
    patient_id: str
    call_type: CallType = CallType.WELLNESS_CHECK
    current_phase: ConversationPhase = ConversationPhase.GREETING
    turn_count: int = 0
    identity_verified: bool = False
    verification_attempts: int = 0
    wellness_scores: list[WellnessScore] = Field(default_factory=list)
    concerns: list[Concern] = Field(default_factory=list)
    message_history: list[dict] = Field(default_factory=list)
    rag_context: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    call_ended: bool = False


class SafetyResult(BaseModel):
    is_safe: bool = True
    blocked_categories: list[str] = Field(default_factory=list)
    groundedness_score: float = 1.0
    prompt_shield_triggered: bool = False


class AlertPayload(BaseModel):
    patient_id: str
    patient_name: str
    alert_type: str
    severity: ConcernSeverity
    summary: str
    concerns: list[Concern] = Field(default_factory=list)
    wellness_scores: list[WellnessScore] = Field(default_factory=list)
    call_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvalResult(BaseModel):
    groundedness: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    fluency: float = 0.0
    safety_score: float = 0.0
    hallucination_flags: list[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())