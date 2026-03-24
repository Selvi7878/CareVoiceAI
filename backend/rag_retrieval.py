"""
CareVoice RAG Retrieval — Azure AI Search with Vector + Keyword Hybrid Search.

Searches the carevoice-protocols index using hybrid search:
1. Generates embedding of the query via Azure OpenAI
2. Runs vector similarity + keyword search simultaneously
3. Returns the most relevant protocol chunks with citation metadata

Called from the orchestrator before the LLM call — results are
injected into the system prompt as grounding context.
"""

from __future__ import annotations

import os
import logging
import rag_retrieval
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "carevoice-protocols")

OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002-2")

API_VERSION_SEARCH = "2024-07-01"
API_VERSION_OPENAI = "2024-10-21"

# Track all RAG retrievals for the citation audit trail
rag_retrieval_log: list[dict] = []


# ─── TOPIC DETECTION ─────────────────────────────────────────────────────────

def _topic_to_query(utterance: str, phase: str) -> str | None:
    """
    Determine if the current utterance needs protocol RAG.
    Returns a natural language query for hybrid search, or None if not needed.
    """
    text = utterance.lower()

    if any(kw in text for kw in ["medication", "medicine", "pill", "forgot", "dose",
                                   "metformin", "lisinopril", "acetaminophen", "side effect"]):
        return "medication management elderly patient forgot adherence protocol"

    if any(kw in text for kw in ["pain", "hurt", "ache", "knee", "sore",
                                   "uncomfortable", "ointment"]):
        return "pain assessment elderly patient chronic acute protocol"

    if any(kw in text for kw in ["fall", "fell", "trip", "stumble", "dizzy",
                                   "lightheaded", "balance"]):
        return "fall prevention response elderly patient protocol"

    if any(kw in text for kw in ["lonely", "sad", "depressed", "anxious", "nobody",
                                   "alone", "don't care", "what's the point", "isolated"]):
        return "emotional wellbeing loneliness depression elderly assessment"

    if any(kw in text for kw in ["eat", "food", "meal", "appetite", "hungry",
                                   "breakfast", "lunch", "dinner", "cooking"]):
        return "nutrition hydration elderly patient meals assessment"

    if any(kw in text for kw in ["confused", "forget", "remember", "what day",
                                   "who are you", "where am i"]):
        return "cognitive screening elderly memory confusion"

    if any(kw in text for kw in ["chest", "breathing", "can't breathe", "emergency",
                                   "help me", "911"]):
        return "emergency escalation caregiver alert critical"

    phase_queries = {
        "physical": "wellness check physical assessment elderly",
        "emotional": "emotional wellbeing assessment elderly protocol",
        "nutrition": "nutrition assessment elderly protocol",
        "cognitive": "cognitive screening elderly protocol",
        "social": "emotional wellbeing loneliness social elderly",
    }
    if phase in phase_queries:
        return phase_queries[phase]

    return None


# ─── EMBEDDING ───────────────────────────────────────────────────────────────

async def _get_embedding(text: str) -> list[float] | None:
    """Generate embedding via Azure OpenAI."""
    if not OPENAI_ENDPOINT or not OPENAI_KEY:
        return None
    url = f"{OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={API_VERSION_OPENAI}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(url,
                headers={"api-key": OPENAI_KEY, "Content-Type": "application/json"},
                json={"input": text})
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
    except Exception as e:
        logger.warning(f"[RAG] Embedding failed: {e}")
        return None


# ─── HYBRID SEARCH (Vector + Keyword) ────────────────────────────────────────

async def search_protocols(utterance: str, phase: str, call_sid: str, turn: int) -> list[dict]:
    """
    Search Azure AI Search using hybrid search (vector + keyword).
    Returns a list of retrieved chunks with citation metadata.
    """
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        return []

    query = _topic_to_query(utterance, phase)
    if not query:
        return []

    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX}/docs/search?api-version={API_VERSION_SEARCH}"

    # Generate embedding for vector search
    query_vector = await _get_embedding(query)

    try:
        # Build hybrid search request
        search_body: dict = {
            "search": query,
            "select": "id,title,category,content,source",
            "top": 2,
            "queryType": "simple",
        }

        # Add vector search if embedding succeeded
        if query_vector:
            search_body["vectorQueries"] = [{
                "kind": "vector",
                "vector": query_vector,
                "fields": "contentVector",
                "k": 2,
            }]

        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(url,
                headers={"api-key": SEARCH_KEY, "Content-Type": "application/json"},
                json=search_body)
            r.raise_for_status()
            results = r.json().get("value", [])

        chunks = []
        for doc in results:
            content = doc.get("content", "")[:800]
            chunk = {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "content": content,
                "source": doc.get("source", ""),
                "search_score": doc.get("@search.score", 0),
                "search_query": query,
                "search_type": "hybrid" if query_vector else "keyword",
            }
            chunks.append(chunk)

        # Log retrieval for audit trail
        retrieval_record = {
            "call_sid": call_sid,
            "turn": turn,
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "search_type": "hybrid" if query_vector else "keyword",
            "utterance_trigger": utterance[:100],
            "results_count": len(chunks),
            "documents_retrieved": [
                {"id": c["id"], "title": c["title"], "source": c["source"], "score": c["search_score"]}
                for c in chunks
            ],
        }
        rag_retrieval_log.append(retrieval_record)

        if chunks:
            logger.info(f"[RAG] Hybrid search: '{query[:40]}...' → {len(chunks)} protocols (type: {'hybrid' if query_vector else 'keyword'})")
        return chunks

    except Exception as e:
        logger.warning(f"[RAG] Search failed: {e}")
        return []


# ─── FORMAT FOR LLM CONTEXT ─────────────────────────────────────────────────

def format_rag_context(chunks: list[dict]) -> str:
    """Format retrieved protocol chunks for injection into the LLM system prompt."""
    if not chunks:
        return ""

    lines = ["\nRELEVANT CARE PROTOCOLS (use these to guide your response — cite them when applicable):"]
    for i, chunk in enumerate(chunks):
        lines.append(f"\n[PROTOCOL {i+1}: {chunk['title']}]")
        lines.append(f"Source: {chunk['source']}")
        lines.append(f"Document ID: {chunk['id']}")
        lines.append(chunk["content"])
        lines.append(f"[END PROTOCOL {i+1}]")

    lines.append("\nWhen your response follows these protocols, you are grounded in verified care guidelines. Do NOT make up procedures not in these documents.")
    return "\n".join(lines)


# ─── API HELPERS ─────────────────────────────────────────────────────────────

def get_rag_log() -> list[dict]:
    return rag_retrieval_log


def get_rag_log_for_call(call_sid: str) -> list[dict]:
    return [r for r in rag_retrieval_log if r["call_sid"] == call_sid]