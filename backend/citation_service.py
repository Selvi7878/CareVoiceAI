"""
Citation Service: tracks source grounding for every AI response.

For the hackathon's "Governed RAG" requirement:
- Every response must cite source fragments
- Every claim must be traceable to a document
- Hallucination risk must be measurable
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

from otel import get_carevoice_tracer

logger = logging.getLogger(__name__)
tracer = get_carevoice_tracer()

# In-memory store for citation records
_citation_store: list[dict] = []


def extract_citations(response: str, rag_context: str, patient_id: str, call_sid: str, turn: int) -> dict:
    """
    Analyze an AI response against the patient's source records.
    Returns a citation record showing which fields were referenced.
    """
    with tracer.start_as_current_span("carevoice.extract_citations") as span:
        span.set_attribute("carevoice.patient_id", patient_id)

        citation = {
            "call_sid": call_sid,
            "patient_id": patient_id,
            "turn": turn,
            "timestamp": datetime.utcnow().isoformat(),
            "response": response,
            "sources_cited": [],
            "grounded": True,
            "ungrounded_claims": [],
        }

        try:
            patient_data = json.loads(rag_context)
        except (json.JSONDecodeError, TypeError):
            citation["grounded"] = False
            citation["ungrounded_claims"].append("No patient context available")
            _citation_store.append(citation)
            return citation

        response_lower = response.lower()

        # Check if response references patient name
        first_name = patient_data.get("firstName", patient_data.get("first_name", ""))
        if first_name and first_name.lower() in response_lower:
            citation["sources_cited"].append({
                "field": "patient.firstName",
                "value": first_name,
                "fragment": f"Patient name: {first_name}",
                "source": "cosmos://carevoice/patients/" + patient_id,
            })

        # Check medication references
        medications = patient_data.get("medications", [])
        for med in medications:
            med_name = med.split("(")[0].strip().split(" ")[0].lower() if isinstance(med, str) else ""
            if med_name and med_name in response_lower:
                citation["sources_cited"].append({
                    "field": "patient.medications",
                    "value": med,
                    "fragment": f"Medication: {med}",
                    "source": "cosmos://carevoice/patients/" + patient_id,
                })

        # Check condition references
        conditions = patient_data.get("conditions", [])
        for cond in conditions:
            keywords = [w.lower() for w in cond.split() if len(w) > 3]
            for kw in keywords:
                if kw in response_lower:
                    citation["sources_cited"].append({
                        "field": "patient.conditions",
                        "value": cond,
                        "fragment": f"Condition: {cond}",
                        "source": "cosmos://carevoice/patients/" + patient_id,
                    })
                    break

        # Check care notes references
        care_notes = patient_data.get("careNotes", patient_data.get("care_notes", ""))
        if care_notes:
            note_keywords = [w.lower() for w in care_notes.split() if len(w) > 4]
            for kw in note_keywords:
                if kw in response_lower and kw not in ["about", "their", "which", "would", "could", "should"]:
                    citation["sources_cited"].append({
                        "field": "patient.careNotes",
                        "value": care_notes[:100] + "...",
                        "fragment": f"Care note keyword: {kw}",
                        "source": "cosmos://carevoice/patients/" + patient_id,
                    })
                    break

        # Check emergency contact references
        contacts = patient_data.get("emergencyContacts", patient_data.get("emergency_contacts", []))
        for contact in contacts:
            name = contact.get("name", "").lower()
            if name and name.split()[0].lower() in response_lower:
                citation["sources_cited"].append({
                    "field": "patient.emergencyContacts",
                    "value": contact.get("name", ""),
                    "fragment": f"Emergency contact: {contact.get('name', '')} ({contact.get('relationship', '')})",
                    "source": "cosmos://carevoice/patients/" + patient_id,
                })

        # Check for potential ungrounded claims
        medical_claims = [
            "you should take", "i recommend", "your diagnosis",
            "you have diabetes", "your blood pressure is",
        ]
        for claim in medical_claims:
            if claim in response_lower:
                citation["grounded"] = False
                citation["ungrounded_claims"].append(f"Potential ungrounded medical claim: '{claim}'")

        # Calculate groundedness score
        if citation["sources_cited"]:
            citation["groundedness_score"] = 1.0 - (len(citation["ungrounded_claims"]) * 0.2)
        else:
            # No citations needed for generic responses like "I'm glad to hear that"
            citation["groundedness_score"] = 0.9 if not citation["ungrounded_claims"] else 0.5

        citation["groundedness_score"] = max(0.0, min(1.0, citation["groundedness_score"]))

        span.set_attribute("carevoice.citations_count", len(citation["sources_cited"]))
        span.set_attribute("carevoice.groundedness", citation["groundedness_score"])

        _citation_store.append(citation)
        return citation


def get_all_citations() -> list[dict]:
    return _citation_store


def get_citations_for_call(call_sid: str) -> list[dict]:
    return [c for c in _citation_store if c["call_sid"] == call_sid]


def get_citation_summary() -> dict:
    """Summary stats for the dashboard."""
    if not _citation_store:
        return {
            "total_responses": 0,
            "total_citations": 0,
            "avg_groundedness": 0.0,
            "ungrounded_count": 0,
            "sources_referenced": [],
        }

    total = len(_citation_store)
    total_citations = sum(len(c["sources_cited"]) for c in _citation_store)
    avg_groundedness = sum(c.get("groundedness_score", 0) for c in _citation_store) / total
    ungrounded = sum(1 for c in _citation_store if not c["grounded"])

    # unique source fields referenced
    all_fields = set()
    for c in _citation_store:
        for s in c["sources_cited"]:
            all_fields.add(s["field"])

    return {
        "total_responses": total,
        "total_citations": total_citations,
        "avg_groundedness": round(avg_groundedness, 3),
        "ungrounded_count": ungrounded,
        "sources_referenced": sorted(list(all_fields)),
    }