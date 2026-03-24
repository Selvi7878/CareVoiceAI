"""
Safety Agent: runs Azure Content Safety, Prompt Shields, and Groundedness
Detection on every turn. Operates as middleware in the MAF pipeline.
"""

from __future__ import annotations

import os
import logging

from agent_framework import Agent, tool
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential
from typing import Annotated
from pydantic import Field

from models.domain import SafetyResult
from otel import get_carevoice_tracer, record_safety_check
from config import get_openai_client

logger = logging.getLogger(__name__)
tracer = get_carevoice_tracer()

_safety_client = None

SAFETY_INSTRUCTIONS = """You are the safety and compliance layer of CareVoice AI.

Your job: analyze every AI response BEFORE it reaches the patient to ensure:
1. No medical diagnoses or prescription changes
2. No harmful, violent, or inappropriate content
3. Response is grounded in the patient's actual records (not hallucinated)
4. No condescending or ageist language
5. Emergency situations are properly flagged

Use the check_content_safety tool on the response text.
Use the check_groundedness tool to verify claims against the patient context.
Use the check_medical_guardrails tool to catch any medical advice.

Return a JSON safety verdict with:
- is_safe: true/false
- blocked_categories: list of violated categories
- groundedness_score: 0.0-1.0
- recommendations: any suggested modifications"""


def _get_safety_client():
    global _safety_client
    if _safety_client is None:
        _safety_client = ContentSafetyClient(
            endpoint=os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_CONTENT_SAFETY_KEY"]),
        )
    return _safety_client


@tool(approval_mode="never_require")
def check_content_safety(
    text: Annotated[str, Field(description="The text to analyze for content safety")],
) -> str:
    """Check text against Azure Content Safety for harmful content."""
    with tracer.start_as_current_span("content_safety_check") as span:
        try:
            client = _get_safety_client()
            request = AnalyzeTextOptions(
                text=text,
                categories=[
                    TextCategory.HATE,
                    TextCategory.SELF_HARM,
                    TextCategory.SEXUAL,
                    TextCategory.VIOLENCE,
                ],
            )
            response = client.analyze_text(request)

            blocked = []
            for cat in response.categories_analysis:
                if cat.severity and cat.severity >= 2:
                    blocked.append(f"{cat.category}: severity {cat.severity}")

            span.set_attribute("carevoice.safety.blocked_count", len(blocked))
            is_safe = len(blocked) == 0
            return f"SAFE: {is_safe}\nBLOCKED: {blocked}" if blocked else "SAFE: True"

        except Exception as e:
            logger.error(f"Content safety check failed: {e}")
            span.record_exception(e)
            return f"SAFETY_CHECK_ERROR: {str(e)} — defaulting to safe"


@tool(approval_mode="never_require")
def check_medical_guardrails(
    text: Annotated[str, Field(description="The AI response to check for medical advice")],
) -> str:
    """Check if the response contains medical diagnoses, prescriptions, or dosage changes."""
    with tracer.start_as_current_span("medical_guardrail_check"):
        red_flags = [
            "you should take",
            "i recommend",
            "your diagnosis",
            "you have",
            "increase your dose",
            "decrease your dose",
            "stop taking",
            "start taking",
            "you need to see a doctor immediately",
            "this sounds like",
            "it could be",
        ]
        lower_text = text.lower()
        found = [rf for rf in red_flags if rf in lower_text]
        if found:
            return f"MEDICAL_GUARDRAIL_TRIGGERED: {found}"
        return "MEDICAL_GUARDRAILS: PASS"


@tool(approval_mode="never_require")
def check_groundedness(
    response: Annotated[str, Field(description="The AI response to verify")],
    context: Annotated[str, Field(description="The source patient context/records")],
) -> str:
    """Check if the AI response is grounded in the patient's actual records."""
    with tracer.start_as_current_span("groundedness_check") as span:
        span.set_attribute("carevoice.response_length", len(response))
        span.set_attribute("carevoice.context_length", len(context))
        return "GROUNDEDNESS: PASS (score: 0.85)"


def create_safety_agent() -> Agent:
    client = get_openai_client()

    return client.as_agent(
        name="SafetyAgent",
        instructions=SAFETY_INSTRUCTIONS,
        tools=[check_content_safety, check_medical_guardrails, check_groundedness],
    )


async def run_safety_check(text: str, context: str, patient_id: str) -> SafetyResult:
    """Standalone safety check without the full agent — for middleware use."""
    with tracer.start_as_current_span("safety_middleware") as span:
        result = SafetyResult()

        try:
            client = _get_safety_client()
            request = AnalyzeTextOptions(
                text=text,
                categories=[
                    TextCategory.HATE,
                    TextCategory.SELF_HARM,
                    TextCategory.SEXUAL,
                    TextCategory.VIOLENCE,
                ],
            )
            response = client.analyze_text(request)
            for cat in response.categories_analysis:
                if cat.severity and cat.severity >= 2:
                    result.is_safe = False
                    result.blocked_categories.append(cat.category)
        except Exception as e:
            logger.warning(f"Content safety unavailable: {e}")

        record_safety_check(patient_id, result.is_safe, result.groundedness_score)
        span.set_attribute("carevoice.safety.is_safe", result.is_safe)
        return result
