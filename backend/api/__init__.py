"""
CareVoice AI — FastAPI REST + WebSocket.
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from workflows import CareVoiceOrchestrator, citations, safety_log
from rag_retrieval import get_rag_log, get_rag_log_for_call
from eval import get_eval_history, get_eval_for_call, evaluate_conversation
from otel import setup_observability

logger = logging.getLogger(__name__)
orchestrator: CareVoiceOrchestrator | None = None


def _hangup_call(call_sid: str):
    """Use Twilio REST API to force-end the call."""
    try:
        from twilio.rest import Client
        client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        client.calls(call_sid).update(status="completed")
        logger.info(f"[HANGUP] Twilio REST API ended call {call_sid}")
    except Exception as e:
        logger.warning(f"[HANGUP] Twilio REST API failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    setup_observability()
    orchestrator = CareVoiceOrchestrator()
    logger.info("CareVoice AI started")
    yield


app = FastAPI(
    title="CareVoice AI — Governed RAG for Elder Care",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


# ─── Twilio TwiML + Call End ─────────────────────────────────────────────────

@app.post("/twiml")
async def twiml(request: Request):
    host = os.environ.get("SERVER_HOST", "localhost")
    port = os.environ.get("SERVER_PORT", "8000")

    ws = (
        f"wss://{host}/ws/call"
        if ("ngrok" in host or "azure" in host)
        else f"wss://{host}:{port}/ws/call"
    )

    action = (
        f"https://{host}/call-end"
        if ("ngrok" in host or "azure" in host)
        else f"http://{host}:{port}/call-end"
    )

    return Response(
        content=(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Response><Connect action="{action}">'
            f'<ConversationRelay url="{ws}" voice="en-US-Journey-O" language="en-US" '
            'transcriptionProvider="google" speechModel="telephony" '
            'interruptible="true" dtmfDetection="true"/>'
            '</Connect></Response>'
        ),
        media_type="application/xml",
    )


@app.post("/call-end")
async def call_end(request: Request):
    return Response(
        content='<?xml version="1.0" encoding="UTF-8"?>\n<Response><Hangup/></Response>',
        media_type="application/xml",
    )


# ─── WebSocket (Core Runtime) ────────────────────────────────────────────────

@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    global orchestrator

    if not orchestrator:
        raise RuntimeError("Orchestrator not initialized")

    await ws.accept()
    call_sid = None

    try:
        while True:
            data = json.loads(await ws.receive_text())
            evt = data.get("type", "")

            if evt == "setup":
                call_sid = data.get("callSid", "unknown")
                pid = data.get("customParameters", {}).get("patientId", "p-001")

                logger.info(f"\n{'='*50}\nCALL START | {call_sid} | {pid}\n{'='*50}")

                greeting = await orchestrator.start_call(call_sid, pid)
                logger.info(f"[CareVoice] {greeting}")

                await ws.send_json({
                    "type": "text",
                    "token": greeting,
                    "last": True
                })

            elif evt == "prompt":
                text = data.get("voicePrompt", "").strip()
                if not text:
                    continue

                logger.info(f"[Patient] {text}")

                response = await orchestrator.handle_utterance(call_sid, text)

                is_end = response.startswith("CALL_END:")
                clean = response.replace("CALL_END:", "").strip()

                # Always send goodbye text first so patient hears it
                if clean:
                    logger.info(f"[CareVoice] {clean}")
                    try:
                        await ws.send_json({
                            "type": "text",
                            "token": clean,
                            "last": True
                        })
                    except Exception as e:
                        logger.warning(f"Failed to send text: {e}")

                if is_end:
                    logger.info(f"CALL END | {call_sid}")

                    # Small delay to let text buffer reach Twilio
                    await asyncio.sleep(0.5)

                    # Send end signal immediately
                    try:
                        await ws.send_json({
                            "type": "end",
                            "handoffData": json.dumps({
                                "reasonCode": "call-complete",
                                "reason": "Wellness check finished"
                            })
                        })
                        logger.info(f"[HANGUP] Sent end signal")
                    except Exception as e:
                        logger.warning(f"[HANGUP] End signal failed: {e}")
                        # Fallback: force hangup via REST API
                        if call_sid and call_sid != "unknown":
                            _hangup_call(call_sid)

                    # Run eval in background
                    state = orchestrator.get_session(call_sid)
                    if state:
                        asyncio.create_task(evaluate_conversation(state))

                    return

            elif evt == "interrupt":
                logger.info("[Interrupt]")

    except WebSocketDisconnect:
        logger.info(f"DISCONNECTED | {call_sid}")

        if call_sid:
            state = orchestrator.get_session(call_sid)
            if state and not state.call_ended:
                state.call_ended = True
                asyncio.create_task(evaluate_conversation(state))

    except Exception as e:
        logger.error(f"WS error: {e}", exc_info=True)


# ─── Sessions ────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def get_sessions():
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")

    return [
        {
            "call_sid": sid,
            "patient_id": s.patient_id,
            "call_type": s.call_type.value,
            "phase": s.current_phase.value,
            "turn_count": s.turn_count,
            "wellness_scores": [w.model_dump() for w in s.wellness_scores],
            "concerns": [c.model_dump() for c in s.concerns],
            "safety_flags": s.safety_flags,
            "started_at": s.started_at.isoformat(),
            "call_ended": s.call_ended,
        }
        for sid, s in orchestrator.get_all_sessions().items()
    ]


@app.get("/api/sessions/{call_sid}")
async def get_session(call_sid: str):
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")

    s = orchestrator.get_session(call_sid)
    if not s:
        raise HTTPException(status_code=404, detail=f"Session {call_sid} not found")

    return {
        "call_sid": s.call_sid,
        "patient_id": s.patient_id,
        "call_type": s.call_type.value,
        "phase": s.current_phase.value,
        "turn_count": s.turn_count,
        "wellness_scores": [w.model_dump() for w in s.wellness_scores],
        "concerns": [c.model_dump() for c in s.concerns],
        "message_history": s.message_history,
        "safety_flags": s.safety_flags,
        "started_at": s.started_at.isoformat(),
        "call_ended": s.call_ended,
    }


# ─── Evaluation ──────────────────────────────────────────────────────────────

@app.get("/api/eval")
async def evals():
    return get_eval_history()


@app.get("/api/eval/{call_sid}")
async def eval_call(call_sid: str):
    r = get_eval_for_call(call_sid)
    if not r:
        raise HTTPException(status_code=404, detail=f"Evaluation for {call_sid} not found")
    return r


# ─── Citations (Governed RAG) ────────────────────────────────────────────────

@app.get("/api/citations")
async def get_citations():
    return citations


@app.get("/api/citations/summary")
async def citation_summary():
    if not citations:
        return {"total_responses": 0, "responses_with_citations": 0,
                "avg_groundedness": 0.0, "ungrounded_responses": 0, "total_source_references": 0}
    return {
        "total_responses": len(citations),
        "responses_with_citations": sum(1 for c in citations if c["sources_cited"]),
        "avg_groundedness": round(sum(c["groundedness_score"] for c in citations) / len(citations), 3),
        "ungrounded_responses": sum(1 for c in citations if c["ungrounded_claims"]),
        "total_source_references": sum(len(c["sources_cited"]) for c in citations),
    }


@app.get("/api/citations/{call_sid}")
async def get_call_citations(call_sid: str):
    return [c for c in citations if c["call_sid"] == call_sid]


# ─── Safety Log ──────────────────────────────────────────────────────────────

@app.get("/api/safety")
async def get_safety_log():
    return safety_log


@app.get("/api/safety/summary")
async def safety_summary():
    if not safety_log:
        return {"total_checks": 0, "passed": 0, "flagged": 0, "avg_groundedness": 0.0}
    return {
        "total_checks": len(safety_log),
        "passed": sum(1 for s in safety_log if s["is_safe"]),
        "flagged": sum(1 for s in safety_log if not s["is_safe"]),
        "avg_groundedness": round(sum(s["groundedness_score"] for s in safety_log) / len(safety_log), 3),
    }


@app.get("/api/safety/{call_sid}")
async def get_safety_for_call(call_sid: str):
    return [s for s in safety_log if s["call_sid"] == call_sid]


# ─── RAG Retrieval Log ───────────────────────────────────────────────────────

@app.get("/api/rag")
async def get_rag_retrieval_log():
    return get_rag_log()


@app.get("/api/rag/{call_sid}")
async def get_rag_for_call(call_sid: str):
    return get_rag_log_for_call(call_sid)


# ─── Agent / Architecture Info ───────────────────────────────────────────────

@app.get("/api/agents")
async def agents():
    return {
        "architecture": "Governed RAG with Citation Tracking",
        "description": "Elder care voice companion with fully traceable, source-grounded responses",
        "framework": "Microsoft Agent Framework",
        "version": "1.0.0rc5",
        "agents": [
            {"name": "Triage", "role": "Classify call type", "status": "active"},
            {"name": "RAG", "role": "Retrieve patient records + protocols", "status": "active"},
            {"name": "Wellness", "role": "Conduct check-in", "status": "active"},
            {"name": "Safety", "role": "Content & medical guardrails", "status": "active"},
            {"name": "Alert", "role": "Escalation & SMS", "status": "active"},
        ],
        "observability": {
            "otel_enabled": True,
            "exporter": "Azure Monitor / OTLP",
            "app_insights": True,
        },
        "evaluation": {
            "sdk": "azure-ai-evaluation",
            "metrics": ["groundedness", "relevance", "coherence", "fluency"],
        },
        "pipeline": [
            {"step": "Identity Verification", "description": "Name confirmation before PHI disclosure", "compliance": "HIPAA"},
            {"step": "RAG Retrieval", "description": "Patient records from Cosmos DB + care protocols from Azure AI Search", "source": "cosmos://carevoice/patients/* + search://carevoice-protocols/*"},
            {"step": "Conversation", "description": "Azure OpenAI GPT-4o-mini with patient data + protocol context in system prompt", "grounding": "All responses grounded in patient record and care protocols"},
            {"step": "Citation Tracking", "description": "Every response analyzed for source references — both patient records and protocol documents", "output": "/api/citations"},
            {"step": "Pre-Response Safety Gate", "description": "Fast keyword scan (<1ms) blocks harmful content before patient hears it", "provider": "Local"},
            {"step": "Async Safety Checks", "description": "Medical guardrails + content safety + groundedness + elder respect + PHI protection", "provider": "Local + Azure AI Content Safety"},
            {"step": "Wellness Scoring", "description": "LLM scoring across 5 dimensions", "dimensions": ["physical", "emotional", "cognitive", "nutrition", "social"]},
            {"step": "LLM Evaluation", "description": "Post-call metrics via azure-ai-evaluation SDK", "metrics": ["groundedness", "relevance", "coherence", "fluency"]},
            {"step": "Alert & Escalation", "description": "SMS via Twilio to caregivers", "provider": "Twilio"},
        ],
    }