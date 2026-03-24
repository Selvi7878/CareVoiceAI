"""
Alert Agent: evaluates wellness scores and concerns at the end of each call,
determines if caregiver notification is needed, and sends SMS via Twilio
or Event Grid alerts for critical escalation.
"""

from __future__ import annotations

import os
import logging

from agent_framework import Agent, tool
from twilio.rest import Client as TwilioClient
from typing import Annotated
from pydantic import Field

from models.domain import AlertPayload, ConcernSeverity
from otel import get_carevoice_tracer
from config import get_openai_client

logger = logging.getLogger(__name__)
tracer = get_carevoice_tracer()

_twilio_client = None

ALERT_INSTRUCTIONS = """You are the alert and escalation component of CareVoice AI.

Your job: after a wellness call ends, analyze the concerns and wellness scores
to determine if caregiver notification is needed.

ESCALATION RULES:
- CRITICAL concerns (fall, chest pain, breathing difficulty): send_sms_alert immediately
- Any wellness dimension score <= 3: send_sms_alert to primary caregiver
- Multiple dimensions scoring 4-5: send_sms_alert with summary
- All scores 6+, no concerns: no alert needed, just log

Use send_sms_alert to notify caregivers.
Use log_alert_decision to record why you did or didn't alert.

Always explain your reasoning."""


def _get_twilio_client():
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(
            os.environ["TWILIO_ACCOUNT_SID"],
            os.environ["TWILIO_AUTH_TOKEN"],
        )
    return _twilio_client


@tool(approval_mode="never_require")
def send_sms_alert(
    phone_number: Annotated[str, Field(description="Caregiver phone number in E.164 format")],
    message: Annotated[str, Field(description="Alert message content")],
) -> str:
    """Send an SMS alert to a caregiver via Twilio."""
    with tracer.start_as_current_span("send_sms_alert") as span:
        span.set_attribute("carevoice.alert.phone", phone_number)
        try:
            client = _get_twilio_client()
            msg = client.messages.create(
                body=message,
                from_=os.environ["TWILIO_PHONE_NUMBER"],
                to=phone_number,
            )
            span.set_attribute("carevoice.alert.sms_sid", msg.sid)
            logger.info(f"SMS alert sent to {phone_number}: {msg.sid}")
            return f"SMS sent successfully (SID: {msg.sid})"
        except Exception as e:
            logger.error(f"SMS alert failed: {e}")
            span.record_exception(e)
            return f"SMS_FAILED: {str(e)}"


@tool(approval_mode="never_require")
def log_alert_decision(
    should_alert: Annotated[bool, Field(description="Whether an alert should be sent")],
    reasoning: Annotated[str, Field(description="Explanation of the alert decision")],
    severity: Annotated[str, Field(description="Overall severity: low, medium, high, critical")],
) -> str:
    """Log the alert decision with reasoning for audit trail."""
    with tracer.start_as_current_span("log_alert_decision") as span:
        span.set_attribute("carevoice.alert.should_alert", should_alert)
        span.set_attribute("carevoice.alert.severity", severity)
        span.set_attribute("carevoice.alert.reasoning", reasoning)
        logger.info(f"Alert decision: alert={should_alert}, severity={severity}, reason={reasoning}")
        return f"Decision logged: alert={should_alert}, severity={severity}"


def create_alert_agent() -> Agent:
    client = get_openai_client()

    return client.as_agent(
        name="AlertAgent",
        instructions=ALERT_INSTRUCTIONS,
        tools=[send_sms_alert, log_alert_decision],
    )
