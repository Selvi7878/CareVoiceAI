"""
Function tools registered on MAF agents via @tool decorator.
Each tool mutates the shared ConversationState and emits OTel spans.
"""

from __future__ import annotations

from typing import Annotated

from agent_framework import tool
from pydantic import Field

from models.domain import (
    WellnessDimension,
    WellnessScore,
    Concern,
    ConcernSeverity,
    ConversationPhase,
)
from otel import get_carevoice_tracer, record_wellness_score

tracer = get_carevoice_tracer()

# Shared state injected at workflow runtime via closure
_state = None


def bind_state(state):
    global _state
    _state = state


@tool(approval_mode="never_require")
def update_wellness_score(
    dimension: Annotated[str, Field(description="Wellness dimension: physical, emotional, cognitive, nutrition, social")],
    score: Annotated[int, Field(description="Score from 1-10, where 10 is best", ge=1, le=10)],
    reasoning: Annotated[str, Field(description="Brief reasoning for the score based on patient's words")],
) -> str:
    """Record a wellness score for a specific dimension based on the conversation."""
    with tracer.start_as_current_span("update_wellness_score") as span:
        dim = WellnessDimension(dimension)
        ws = WellnessScore(dimension=dim, score=score, reasoning=reasoning)
        _state.wellness_scores.append(ws)
        record_wellness_score(_state.patient_id, dimension, score)
        span.set_attribute("carevoice.dimension", dimension)
        span.set_attribute("carevoice.score", score)
        return f"Recorded {dimension} score: {score}/10"


@tool(approval_mode="never_require")
def log_concern(
    category: Annotated[str, Field(description="Concern category: medication, mobility, pain, mood, memory, nutrition, isolation, safety")],
    severity: Annotated[str, Field(description="Severity: low, medium, high, critical")],
    description: Annotated[str, Field(description="What the patient said or implied")],
    suggested_action: Annotated[str, Field(description="Recommended follow-up action")],
) -> str:
    """Log a health or safety concern detected during conversation."""
    with tracer.start_as_current_span("log_concern") as span:
        concern = Concern(
            category=category,
            severity=ConcernSeverity(severity),
            description=description,
            suggested_action=suggested_action,
        )
        _state.concerns.append(concern)
        span.set_attribute("carevoice.concern_category", category)
        span.set_attribute("carevoice.concern_severity", severity)
        return f"Logged {severity} concern: {category}"


@tool(approval_mode="never_require")
def advance_phase() -> str:
    """Advance to the next conversation phase after completing the current one."""
    phases = list(ConversationPhase)
    current_idx = phases.index(_state.current_phase)
    if current_idx < len(phases) - 1:
        _state.current_phase = phases[current_idx + 1]
        return f"Advanced to phase: {_state.current_phase.value}"
    return "Already in closing phase"


@tool(approval_mode="never_require")
def end_call(
    summary: Annotated[str, Field(description="Brief summary of the call")],
) -> str:
    """End the wellness check call with a summary."""
    with tracer.start_as_current_span("end_call") as span:
        span.set_attribute("carevoice.call_summary", summary)
        return f"CALL_END: {summary}"
