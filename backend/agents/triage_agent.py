"""
Triage Agent: classifies the incoming call into one of three flows.
Runs once at the start of each call using the caller's first utterance.
"""

from __future__ import annotations

import os

from agent_framework import Agent, tool

from models.domain import CallType
from config import get_openai_client


TRIAGE_INSTRUCTIONS = """You are the triage component of CareVoice AI, an elder care wellness system.

Your job: classify the caller's intent into exactly one of these categories:
- wellness_check: The patient is receiving a scheduled wellness call, or is calling in for a routine check-in.
- emergency: The patient reports an emergency, severe pain, a fall, chest pain, difficulty breathing, or any life-threatening situation.
- general_inquiry: The patient has a general question about their care, medications, appointments, or wants to speak with someone.

Respond with ONLY the classification and a one-sentence reason. Format:
CLASSIFICATION: <wellness_check|emergency|general_inquiry>
REASON: <brief explanation>

If the input is a greeting like "hello" or "hi" with no other context, classify as wellness_check since this is likely a scheduled call."""


@tool(approval_mode="never_require")
def classify_call(
    classification: str,
    reason: str,
) -> str:
    """Submit the call classification result."""
    return f"CLASSIFICATION: {classification}\nREASON: {reason}"


def create_triage_agent() -> Agent:
    client = get_openai_client()

    return client.as_agent(
        name="TriageAgent",
        instructions=TRIAGE_INSTRUCTIONS,
        tools=[classify_call],
    )


def parse_triage_result(result: str) -> CallType:
    text = str(result).lower()
    if "emergency" in text:
        return CallType.EMERGENCY
    if "general_inquiry" in text:
        return CallType.GENERAL_INQUIRY
    return CallType.WELLNESS_CHECK
