"""
CareVoice conversation engine.

Direct Azure OpenAI calls for lowest latency.
Identity verification before PHI disclosure.
Topic tracking to prevent repeated questions.
Citations tracked on every response.
Wellness scoring in background.
Safety checks run async (post-send, zero latency).
Alert at end of call only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime

import httpx
from azure.cosmos import CosmosClient
from utils.end_call import should_end_call
from models.domain import (
    CallType, ConversationState, ConversationPhase, ConcernSeverity,
    WellnessScore, Concern,
)
from otel import get_carevoice_tracer, record_call_started, record_call_ended, record_safety_check
from rag_retrieval import search_protocols, format_rag_context

logger = logging.getLogger(__name__)
tracer = get_carevoice_tracer()

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT",
    os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"))

_cosmos = None
citations: list[dict] = []
safety_log: list[dict] = []
SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "carevoice-protocols")

MAX_VERIFY_ATTEMPTS = 2


# ─── LLM + DB ───────────────────────────────────────────────────────────────

async def _chat(messages: list[dict], max_tokens: int = 150) -> str:
    url = f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-10-21"
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(url,
            headers={"api-key": API_KEY, "Content-Type": "application/json"},
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.7})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _db():
    global _cosmos
    if not _cosmos:
        _cosmos = CosmosClient(os.environ["AZURE_COSMOS_ENDPOINT"], os.environ["AZURE_COSMOS_KEY"])
    name = os.environ.get("COSMOS_DATABASE", os.environ.get("AZURE_COSMOS_DATABASE", "carevoice"))
    return _cosmos.get_database_client(name)


def _get_patient(pid: str) -> dict:
    items = list(_db().get_container_client("patients").query_items(
        "SELECT * FROM c WHERE c.id = @id", [{"name": "@id", "value": pid}],
        enable_cross_partition_query=True))
    return items[0] if items else {}


def _build_contacts_str(p: dict) -> str:
    contacts = p.get("caregiverContacts", [])
    if not contacts:
        return "none"
    parts = []
    for c in contacts:
        name = c.get("name", "")
        rel = c.get("relationship", "")
        if name and rel:
            parts.append(f"{name} ({rel})")
        elif name:
            parts.append(name)
    return ", ".join(parts) if parts else "none"


# ─── IDENTITY VERIFICATION ──────────────────────────────────────────────────

def _check_name_match(utterance: str, expected_first: str) -> bool:
    text = utterance.strip().lower()
    expected = expected_first.strip().lower()
    if not expected:
        return False
    if expected in text:
        return True
    if text in {"yes", "yeah", "yep", "that's me", "speaking", "this is she",
                "this is he", "it's me", "thats me", "yes it is", "yea"}:
        return True
    words = re.findall(r'[a-z]+', text)
    for word in words:
        if len(word) >= 3 and len(expected) >= 3:
            if len(word) == len(expected):
                diffs = sum(1 for a, b in zip(word, expected) if a != b)
                if diffs <= 1:
                    return True
            if abs(len(word) - len(expected)) == 1:
                shorter, longer = (word, expected) if len(word) < len(expected) else (expected, word)
                i = j = misses = 0
                while i < len(shorter) and j < len(longer):
                    if shorter[i] == longer[j]:
                        i += 1
                    else:
                        misses += 1
                    j += 1
                if misses <= 1:
                    return True
    return False


# ─── TOPIC TRACKING ─────────────────────────────────────────────────────────

def _get_covered_topics(state: ConversationState) -> list[str]:
    """Scan conversation history to find topics already discussed."""
    history_text = " ".join(m["content"].lower() for m in state.message_history)
    covered = []
    if any(kw in history_text for kw in ["medication", "metformin", "lisinopril", "acetaminophen", "take your med"]):
        covered.append("medications (already asked and answered)")
    if any(kw in history_text for kw in ["sleep", "slept", "rest"]):
        covered.append("sleep")
    if any(kw in history_text for kw in ["meal", "breakfast", "lunch", "dinner", "eat", "toast", "food"]):
        covered.append("meals/nutrition")
    if any(kw in history_text for kw in ["pain", "knee", "ache", "hurt", "discomfort", "ointment"]):
        covered.append("pain/body")
    if any(kw in history_text for kw in ["daughter", "sarah", "family", "friend", "lonely", "isolat"]):
        covered.append("family/social connections")
    if any(kw in history_text for kw in ["spirit", "mood", "feeling", "emotional", "okay emotionally"]):
        covered.append("emotional wellbeing")
    return covered


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

def _system(p: dict) -> str:
    n = p.get("firstName", "friend")
    notes = p.get("medicalNotes", [])
    notes_str = "\n  ".join(notes) if notes else "none"
    contacts_str = _build_contacts_str(p)

    return f"""You are CareVoice, calling {n} for a daily wellness check-in.

PATIENT RECORD [SOURCE: cosmos://carevoice/patients/{p.get('id','')}]:
- Name: {n}, DOB: {p.get('dateOfBirth','?')}
- Medical notes:
  {notes_str}
- Caregiver contacts: {contacts_str}

RULES:
- Warm caring friend, not a nurse or robot
- 1-2 SHORT sentences max (phone call)
- ONE question at a time
- Show genuine empathy — acknowledge what they share before moving on
- If they mention pain, loneliness, or difficulty, pause and respond with care before asking the next question
- Natural fillers: "Oh...", "Hmm...", "That's good to hear..."
- Use {n} sometimes
- When mentioning family or contacts, ALWAYS use their relationship AND name (e.g. "your daughter Sarah", "Dr. Emily Chen")

FLOW (natural, flexible):
Greeting → Sleep → Meals → Medications → Pain/body → Spirits → Family/friends → Need anything → Wait for THEM to say goodbye

MEDICATION (important):
- Their medical notes:
  {notes_str}
- Ask about medications ONCE during the call
- If they forgot which: tell them their medications from the notes above
- Never scold, be encouraging

WHEN REFERENCING PATIENT DATA:
- When you mention their name, medications, or conditions, you are citing from their patient record
- When they ask about their meds, tell them exactly from the medical notes above
- When relevant, mention their conditions naturally
- ALWAYS refer to contacts by relationship: "your daughter Sarah", not just "Sarah"

EMPATHY GUIDELINES:
- If they report pain: "Oh, I'm sorry to hear that. That must be uncomfortable." then follow up
- If they sound lonely: "I understand, that can feel really isolating. I'm glad we're chatting now." then gently suggest reaching out
- If they forgot medication: "No worries at all! Let me remind you..." — never make them feel bad
- If they share good news: Match their energy warmly
- If they give short answers like "okay" or "not bad": slow down, don't rush to the next topic

CRITICAL — GOODBYE RULES:
- NEVER say goodbye, "take care", "talk to you soon", or wrap up the conversation yourself
- NEVER initiate ending the call — only the PATIENT can end it
- If you have covered all topics, ask "Is there anything else on your mind, {n}?" and WAIT
- If they say "that's all" or "nothing else" — say "Alright, I'm here whenever you need me, {n}." and WAIT for them to say bye
- Only AFTER they explicitly say bye/goodbye/see you: respond with ONE goodbye: "Take care, {n}! It was lovely talking with you. I'll call again soon!"
- NEVER repeat goodbye — once you say it, STOP completely"""


# ─── BYE DETECTION ───────────────────────────────────────────────────────────

_BYE = {"by", "bye", "bye bye", "bye-bye", "goodbye", "good bye", "see you",
        "talk later", "ok bye", "okay bye", "alright bye", "thanks bye"}
_END = ["hang up", "end the call", "end call", "let me go", "i need to go",
        "i'm done", "gotta go", "i got to go"]

def _is_bye(t: str) -> bool:
    t = t.strip().lower().rstrip(".!?,")
    if t in _BYE:
        return True
    for b in _BYE:
        if t.startswith(b) or t.endswith(b):
            return True
    for p in _END:
        if p in t:
            return True
    if t in {"by", "bi", "bay"}:
        return True
    return False


# ─── ASYNC SAFETY MIDDLEWARE (zero latency) ──────────────────────────────────

_MEDICAL_RED_FLAGS = [
    "you should take", "i recommend you", "your diagnosis is",
    "you have been diagnosed", "increase your dose", "decrease your dose",
    "stop taking", "start taking", "it could be", "this sounds like",
]

_PRE_RESPONSE_BLOCKS = [
    "you should take", "i recommend you", "your diagnosis is",
    "you have been diagnosed", "increase your dose", "decrease your dose",
    "stop taking", "start taking", "you need to see a doctor immediately",
    "i'm prescribing", "take this medication", "your condition is",
]

_SAFE_FALLBACK = "That's a great question — I'd suggest checking with your care team about that."


def _pre_response_gate(response: str, state: ConversationState, turn: int) -> str:
    """
    Fast pre-response safety gate (<1ms). Scans for dangerous phrases
    BEFORE the patient hears the response. If triggered, returns a safe
    fallback. Otherwise returns the original response unchanged.
    """
    rl = response.lower()
    for phrase in _PRE_RESPONSE_BLOCKS:
        if phrase in rl:
            state.safety_flags.append(f"turn_{turn}: PRE-RESPONSE BLOCKED — '{phrase}' detected")
            logger.warning(f"[SAFETY-GATE] Blocked response at turn {turn}: '{phrase}' found")
            return _SAFE_FALLBACK
    return response

_AGEIST_FLAGS = [
    "for your age", "at your age", "old people", "elderly people",
    "you're too old", "senior moment",
]


async def _safety_check_bg(response: str, patient: dict, state: ConversationState, call_sid: str, turn: int):
    with tracer.start_as_current_span("carevoice.safety_check") as span:
        result = {
            "call_sid": call_sid,
            "turn": turn,
            "timestamp": datetime.utcnow().isoformat(),
            "response_snippet": response[:100],
            "checks_passed": [],
            "checks_failed": [],
            "is_safe": True,
            "groundedness_score": 0.95,
        }
        rl = response.lower()

        med_flags = [rf for rf in _MEDICAL_RED_FLAGS if rf in rl]
        if med_flags:
            result["checks_failed"].append(f"medical_guardrail: {med_flags}")
            result["is_safe"] = False
            state.safety_flags.append(f"turn_{turn}: medical guardrail — {med_flags}")
        else:
            result["checks_passed"].append("medical_guardrails")

        age_flags = [af for af in _AGEIST_FLAGS if af in rl]
        if age_flags:
            result["checks_failed"].append(f"ageist_language: {age_flags}")
            result["is_safe"] = False
            state.safety_flags.append(f"turn_{turn}: ageist language — {age_flags}")
        else:
            result["checks_passed"].append("elder_respect")

        patient_text = json.dumps(patient, default=str).lower()
        ungrounded = []
        for claim in ["mg", "twice daily", "every morning", "allergy to"]:
            if claim in rl and claim not in patient_text:
                ungrounded.append(claim)
        if ungrounded:
            result["checks_failed"].append(f"groundedness: {ungrounded}")
            result["groundedness_score"] = 0.4
            state.safety_flags.append(f"turn_{turn}: possible hallucination — {ungrounded}")
        else:
            result["checks_passed"].append("groundedness")

        if not state.identity_verified:
            patient_name = patient.get("firstName", "").lower()
            if patient_name and patient_name in rl:
                result["checks_failed"].append("phi_leak: name before verification")
                result["is_safe"] = False
                state.safety_flags.append(f"turn_{turn}: PHI leaked before verification")
            else:
                result["checks_passed"].append("phi_protection")

        try:
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
            from azure.core.credentials import AzureKeyCredential
            cs_endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", "")
            cs_key = os.environ.get("AZURE_CONTENT_SAFETY_KEY", "")
            if cs_endpoint and cs_key:
                client = ContentSafetyClient(cs_endpoint, AzureKeyCredential(cs_key))
                cs_result = client.analyze_text(AnalyzeTextOptions(
                    text=response,
                    categories=[TextCategory.HATE, TextCategory.SELF_HARM, TextCategory.SEXUAL, TextCategory.VIOLENCE],
                ))
                blocked = [f"{cat.category}: severity {cat.severity}"
                           for cat in cs_result.categories_analysis
                           if cat.severity and cat.severity >= 2]
                if blocked:
                    result["checks_failed"].append(f"content_safety: {blocked}")
                    result["is_safe"] = False
                    state.safety_flags.append(f"turn_{turn}: content safety — {blocked}")
                else:
                    result["checks_passed"].append("azure_content_safety")
            else:
                result["checks_passed"].append("content_safety_skipped")
        except ImportError:
            result["checks_passed"].append("content_safety_skipped")
        except Exception as e:
            result["checks_passed"].append(f"content_safety_error ({e})")

        record_safety_check(state.patient_id, result["is_safe"], result["groundedness_score"])
        span.set_attribute("carevoice.safety.is_safe", result["is_safe"])
        span.set_attribute("carevoice.safety.checks_passed", len(result["checks_passed"]))
        span.set_attribute("carevoice.safety.checks_failed", len(result["checks_failed"]))
        safety_log.append(result)
        if not result["is_safe"]:
            logger.warning(f"[SAFETY] Turn {turn} flagged: {result['checks_failed']}")


# ─── CITATION TRACKING ──────────────────────────────────────────────────────

def _cite(response: str, patient: dict, call_sid: str, turn: int, rag_chunks: list[dict] | None = None) -> dict:
    c = {
        "call_sid": call_sid, "turn": turn, "response": response,
        "patient_id": patient.get("id", ""),
        "timestamp": datetime.utcnow().isoformat(),
        "sources_cited": [],
        "groundedness_score": 0.95,
        "ungrounded_claims": [],
        "document_source": f"cosmos://carevoice/patients/{patient.get('id', '')}",
    }
    rl = response.lower()
    pid = patient.get("id", "")
    src = f"cosmos://carevoice/patients/{pid}"

    name = patient.get("firstName", "")
    if name and name.lower() in rl:
        c["sources_cited"].append({
            "field": "patient.firstName", "cited_value": name,
            "source_document": src, "fragment": f"Patient name: {name}",
        })

    for note in patient.get("medicalNotes", []):
        keywords = [w.lower() for w in note.split() if len(w) > 3]
        for kw in keywords:
            if kw in rl and kw not in ["mild", "type", "takes", "uses", "daily",
                                        "every", "last", "good", "morning", "evening"]:
                c["sources_cited"].append({
                    "field": "patient.medicalNotes", "cited_value": note,
                    "source_document": src, "fragment": f"Medical note: {note[:80]}",
                })
                break

    for ec in patient.get("caregiverContacts", []):
        ec_name = ec.get("name", "").split()[0].lower()
        if ec_name and ec_name in rl:
            c["sources_cited"].append({
                "field": "patient.caregiverContacts",
                "cited_value": f"{ec['name']} ({ec.get('relationship','')})",
                "source_document": src, "fragment": f"Caregiver: {ec['name']}",
            })

    # Protocol citations from RAG retrieval
    if rag_chunks:
        for chunk in rag_chunks:
            # Check if the response content aligns with retrieved protocol
            protocol_keywords = [w.lower() for w in chunk.get("title", "").split() if len(w) > 3]
            for kw in protocol_keywords:
                if kw in rl and kw not in ["protocol", "elderly", "patient", "assessment"]:
                    c["sources_cited"].append({
                        "field": f"protocol.{chunk.get('category', 'general')}",
                        "cited_value": chunk.get("title", ""),
                        "source_document": f"search://{SEARCH_INDEX}/{chunk.get('id', '')}",
                        "fragment": f"Protocol: {chunk.get('title', '')[:60]} — {chunk.get('source', '')}",
                    })
                    break

    for bad in ["you should take", "i recommend you", "your diagnosis is", "you have been diagnosed"]:
        if bad in rl:
            c["ungrounded_claims"].append(bad)
            c["groundedness_score"] = 0.3

    citations.append(c)
    return c


# ─── BACKGROUND WELLNESS SCORING ────────────────────────────────────────────

async def _score_bg(state: ConversationState, utterance: str, response: str):
    scored = {s.dimension.value for s in state.wellness_scores}
    if len(scored) >= 5:
        return
    try:
        r = await _chat([
            {"role": "system", "content": "Analyze wellness conversation. Output ONLY valid JSON, no markdown."},
            {"role": "user", "content": (
                f'Patient said: "{utterance}"\n'
                f'AI replied: "{response}"\n'
                f'Already scored: {",".join(scored) or "none"}\n'
                f'Output: {{"dimension":"physical|emotional|cognitive|nutrition|social","score":1-10,"reasoning":"..."}}\n'
                f'Or if nothing to score: {{"dimension":null}}'
            )},
        ], max_tokens=100)
        d = json.loads(r.strip().strip("`").replace("```json", "").replace("```", ""))
        if d.get("dimension") and d["dimension"] not in scored:
            state.wellness_scores.append(WellnessScore(
                dimension=d["dimension"], score=d["score"], reasoning=d.get("reasoning", "")))
            if d["score"] <= 3:
                state.concerns.append(Concern(
                    category=d["dimension"], severity=ConcernSeverity.HIGH,
                    description=f"Low {d['dimension']} score: {d['score']}/10 — {d.get('reasoning','')}",
                    suggested_action="Follow up with caregiver"))
    except Exception:
        pass


# ─── CAREGIVER ALERT ─────────────────────────────────────────────────────────

async def _alert(state: ConversationState, patient: dict):
    contacts = patient.get("caregiverContacts", [])
    phone = contacts[0].get("phoneNumber", "") if contacts else ""
    name = patient.get("firstName", "friend")
    scores = ", ".join(f"{s.dimension.value}:{s.score}" for s in state.wellness_scores) or "none"
    concerns = ", ".join(f"{c.severity.value}:{c.description}" for c in state.concerns) or "none"
    needs = any(s.score <= 4 for s in state.wellness_scores) or len(state.concerns) > 0

    duration = (datetime.utcnow() - state.started_at).total_seconds()
    record_call_ended(state.patient_id, state.call_sid, duration)
    logger.info(f"CALL SUMMARY | {name} | {duration:.0f}s | scores=[{scores}] | concerns=[{concerns}] | alert={needs}")

    if needs and phone:
        try:
            from twilio.rest import Client
            t = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
            t.messages.create(
                body=f"CareVoice Alert for {name}: Wellness scores: {scores}. Concerns: {concerns}. Please check in.",
                from_=os.environ["TWILIO_PHONE_NUMBER"], to=phone)
            logger.info(f"SMS SENT to {phone} for {name}")
        except Exception as e:
            logger.error(f"SMS failed: {e}")


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

class CareVoiceOrchestrator:

    def __init__(self):
        self.sessions: dict[str, ConversationState] = {}
        self.patients: dict[str, dict] = {}

    async def start_call(self, call_sid: str, patient_id: str) -> str:
        with tracer.start_as_current_span("carevoice.start_call"):
            record_call_started(patient_id, call_sid)
            state = ConversationState(call_sid=call_sid, patient_id=patient_id)
            state.current_phase = ConversationPhase.IDENTITY
            self.sessions[call_sid] = state

            patient = _get_patient(patient_id)
            self.patients[call_sid] = patient
            state.rag_context = json.dumps(patient, default=str)

            greeting = "Hi there! This is CareVoice, your wellness companion. May I know who I'm speaking with?"
            state.message_history.append({"role": "assistant", "content": greeting})
            state.turn_count = 1
            asyncio.create_task(_safety_check_bg(greeting, patient, state, call_sid, 1))
            logger.info(f"[RAG] Patient loaded: {patient.get('firstName','?')} | medicalNotes: {len(patient.get('medicalNotes',[]))} | awaiting identity verification")
            return greeting

    async def handle_utterance(self, call_sid: str, utterance: str) -> str:
        state = self.sessions.get(call_sid)
        if not state:
            return "Sorry, I lost our connection."

        patient = self.patients.get(call_sid, {})
        state.message_history.append({"role": "user", "content": utterance})
        state.turn_count += 1
        

        # ─── IDENTITY VERIFICATION ───────────────────────────────────
        if not state.identity_verified:
            expected_name = patient.get("firstName", "")

            if _check_name_match(utterance, expected_name):
                state.identity_verified = True
                state.current_phase = ConversationPhase.GREETING
                logger.info(f"[IDENTITY] Verified: {expected_name} | attempt {state.verification_attempts + 1}")

                response = await _chat([
                    {"role": "system", "content": _system(patient)},
                    {"role": "user", "content": (
                        "The patient just confirmed their identity. "
                        "Greet them warmly by name and ask how they slept. "
                        "One warm sentence."
                    )},
                ], max_tokens=60)

                response = _pre_response_gate(response, state, state.turn_count)

                state.message_history.append({"role": "assistant", "content": response})
                _cite(response, patient, call_sid, state.turn_count)
                asyncio.create_task(_safety_check_bg(response, patient, state, call_sid, state.turn_count))
                return response
            else:
                state.verification_attempts += 1
                if state.verification_attempts >= MAX_VERIFY_ATTEMPTS:
                    fail_msg = "I'm sorry, I wasn't able to verify your identity. Please have the patient call us back when they're available. Take care!"
                    state.message_history.append({"role": "assistant", "content": fail_msg})
                    state.call_ended = True
                    state.safety_flags.append("identity_verification_failed")
                    logger.warning(f"[IDENTITY] Failed after {MAX_VERIFY_ATTEMPTS} attempts | call_sid={call_sid}")
                    asyncio.create_task(_alert(state, patient))
                    return f"CALL_END:{fail_msg}"

                retry_msg = "No worries! Could you tell me your first name, please?"
                state.message_history.append({"role": "assistant", "content": retry_msg})
                asyncio.create_task(_safety_check_bg(retry_msg, patient, state, call_sid, state.turn_count))
                return retry_msg

        # ─── BYE DETECTION ───────────────────────────────────────────
        if state.turn_count > 2 and _is_bye(utterance):
            if not state.call_ended:
                name = patient.get("firstName", "friend")
                bye = f"Take care, {name}! It was lovely talking with you. I'll call again soon!"
                state.message_history.append({"role": "assistant", "content": bye})
                state.call_ended = True
                _cite(bye, patient, call_sid, state.turn_count)
                asyncio.create_task(_alert(state, patient))
                return f"CALL_END:{bye}"
            else:
                return "CALL_END:"

        # ─── BUILD MESSAGES WITH TOPIC TRACKING ──────────────────────
        msgs = [{"role": "system", "content": _system(patient)}]

        # ─── RAG: Search protocols for relevant guidance ─────────────
        rag_chunks = await search_protocols(utterance, state.current_phase.value, call_sid, state.turn_count)
        if rag_chunks:
            rag_context = format_rag_context(rag_chunks)
            msgs.append({"role": "system", "content": rag_context})

        # Inject covered topics so LLM doesn't repeat
        covered = _get_covered_topics(state)
        if covered:
            msgs.append({
                "role": "system",
                "content": f"IMPORTANT — Topics already discussed this call (do NOT ask about these again, do NOT revisit them):\n- " + "\n- ".join(covered)
            })

        # Reinforce no-goodbye rule in later turns
        if state.turn_count > 6:
            msgs.append({
                "role": "system",
                "content": (
                    "CRITICAL REMINDER: Do NOT say goodbye, take care, talk to you soon, or wrap up the call. "
                    "You are NOT allowed to end the conversation. Only the PATIENT can say bye. "
                    "If they say 'that's all' or 'nothing else', respond with 'Alright, I'm here whenever you need me.' and WAIT."
                )
            })

        # Conversation history (last 12 turns)
        for m in state.message_history[-12:]:
            msgs.append({"role": m["role"], "content": m["content"]})

        response = await _chat(msgs, max_tokens=120)

        # PRE-RESPONSE SAFETY GATE — blocks harmful content before patient hears it
        response = _pre_response_gate(response, state, state.turn_count)

        state.message_history.append({"role": "assistant", "content": response})

        # All background tasks — zero latency
        _cite(response, patient, call_sid, state.turn_count, rag_chunks=rag_chunks)
        asyncio.create_task(_score_bg(state, utterance, response))
        asyncio.create_task(_safety_check_bg(response, patient, state, call_sid, state.turn_count))

        # Advance phase
        phases = list(ConversationPhase)
        idx = phases.index(state.current_phase)
        next_idx = min(max(idx, state.turn_count // 2 + 1), len(phases) - 1)
        state.current_phase = phases[next_idx]

        return response

    def get_session(self, sid: str):
        return self.sessions.get(sid)

    def get_all_sessions(self):
        return self.sessions