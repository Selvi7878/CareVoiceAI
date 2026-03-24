"""
RAG Agent: retrieves patient medical records, medications, care protocols,
and prior conversation history from Cosmos DB to build grounded context.
"""

from __future__ import annotations

import os
import json
import logging

from agent_framework import Agent, tool
from azure.cosmos import CosmosClient
from typing import Annotated
from pydantic import Field

from otel import get_carevoice_tracer
from config import get_openai_client, get_cosmos_database, get_cosmos_container

logger = logging.getLogger(__name__)
tracer = get_carevoice_tracer()

RAG_INSTRUCTIONS = """You are the RAG (retrieval-augmented generation) component of CareVoice AI.

Your job: given a patient ID, retrieve their records and build a grounded context summary
that the Wellness Agent will use during the conversation.

Use the retrieve_patient_context tool to fetch the patient's data, then synthesize it into
a structured context block with these sections:
- PATIENT PROFILE: name, age, conditions
- MEDICATIONS: current medications and schedule
- CARE NOTES: relevant care instructions
- RECENT HISTORY: summary of last wellness check if available
- CONVERSATION GUIDELINES: any special communication needs

Be factual. Only include information that exists in the retrieved records."""


_cosmos_client = None


def _get_cosmos_client():
    global _cosmos_client
    if _cosmos_client is None:
        _cosmos_client = CosmosClient(
            url=os.environ["AZURE_COSMOS_ENDPOINT"],
            credential=os.environ["AZURE_COSMOS_KEY"],
        )
    return _cosmos_client


@tool(approval_mode="never_require")
def retrieve_patient_context(
    patient_id: Annotated[str, Field(description="The patient's unique identifier")],
) -> str:
    """Retrieve patient records from Cosmos DB including medical history, medications, and care notes."""
    with tracer.start_as_current_span("cosmos_patient_lookup") as span:
        span.set_attribute("carevoice.patient_id", patient_id)
        try:
            client = _get_cosmos_client()
            db = client.get_database_client(get_cosmos_database())
            container = db.get_container_client(get_cosmos_container())

            query = "SELECT * FROM c WHERE c.id = @pid"
            params = [{"name": "@pid", "value": patient_id}]
            items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))

            if not items:
                span.set_attribute("carevoice.patient_found", False)
                return f"No patient found with ID: {patient_id}"

            patient = items[0]
            span.set_attribute("carevoice.patient_found", True)
            return json.dumps(patient, indent=2, default=str)

        except Exception as e:
            logger.error(f"Cosmos DB lookup failed: {e}")
            span.record_exception(e)
            return f"Error retrieving patient data: {str(e)}"


@tool(approval_mode="never_require")
def retrieve_conversation_history(
    patient_id: Annotated[str, Field(description="The patient's unique identifier")],
    limit: Annotated[int, Field(description="Number of recent conversations to retrieve", ge=1, le=10)] = 3,
) -> str:
    """Retrieve recent conversation history for the patient."""
    with tracer.start_as_current_span("cosmos_history_lookup") as span:
        span.set_attribute("carevoice.patient_id", patient_id)
        try:
            client = _get_cosmos_client()
            db = client.get_database_client(get_cosmos_database())
            container = db.get_container_client("conversations")

            query = (
                "SELECT TOP @limit * FROM c WHERE c.patientId = @pid "
                "ORDER BY c.timestamp DESC"
            )
            params = [
                {"name": "@pid", "value": patient_id},
                {"name": "@limit", "value": limit},
            ]
            items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
            span.set_attribute("carevoice.history_count", len(items))
            return json.dumps(items, indent=2, default=str)

        except Exception as e:
            logger.error(f"History lookup failed: {e}")
            span.record_exception(e)
            return f"No conversation history available: {str(e)}"


def create_rag_agent() -> Agent:
    client = get_openai_client()

    return client.as_agent(
        name="RAGAgent",
        instructions=RAG_INSTRUCTIONS,
        tools=[retrieve_patient_context, retrieve_conversation_history],
    )
