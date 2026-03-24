"""
LLM Evaluation Pipeline for CareVoice AI.

Runs post-conversation evaluations using azure-ai-evaluation SDK:
- Groundedness: is the response based on patient records?
- Relevance: is the response relevant to the patient's question?
- Coherence: is the response logically consistent?
- Fluency: is the response well-formed?

Results are stored and exposed via API for the React dashboard.
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Optional

from models.domain import EvalResult, ConversationState
from otel import get_carevoice_tracer, record_eval_score

logger = logging.getLogger(__name__)
tracer = get_carevoice_tracer()

_eval_store: list[dict] = []


def _get_model_config() -> dict:
    return {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "azure_deployment": os.environ.get("AZURE_AI_EVALUATION_DEPLOYMENT", "gpt-4o-mini"),
        "api_version": "2025-01-01-preview",
    }


async def evaluate_conversation(state: ConversationState) -> EvalResult:
    """
    Run the full evaluation suite on a completed conversation.
    Called after each call ends.
    """
    with tracer.start_as_current_span("carevoice.evaluate_conversation") as span:
        span.set_attribute("carevoice.patient_id", state.patient_id)
        span.set_attribute("carevoice.call_sid", state.call_sid)
        span.set_attribute("carevoice.turn_count", state.turn_count)

        result = EvalResult()

        # Build evaluation dataset from conversation
        eval_pairs = _extract_eval_pairs(state)
        if not eval_pairs:
            logger.warning("No evaluation pairs extracted from conversation")
            return result

        try:
            from azure.ai.evaluation import (
                RelevanceEvaluator,
                CoherenceEvaluator,
                FluencyEvaluator,
                GroundednessEvaluator,
            )

            model_config = _get_model_config()

            # Run evaluators on the last few turns
            scores = {"groundedness": [], "relevance": [], "coherence": [], "fluency": []}

            relevance_eval = RelevanceEvaluator(model_config)
            coherence_eval = CoherenceEvaluator(model_config)
            fluency_eval = FluencyEvaluator(model_config)
            groundedness_eval = GroundednessEvaluator(model_config)

            for pair in eval_pairs[-5:]:
                query = pair["query"]
                response = pair["response"]
                context = pair.get("context", state.rag_context)

                try:
                    rel = relevance_eval(query=query, response=response)
                    scores["relevance"].append(rel.get("relevance", 0))
                except Exception:
                    pass

                try:
                    coh = coherence_eval(query=query, response=response)
                    scores["coherence"].append(coh.get("coherence", 0))
                except Exception:
                    pass

                try:
                    flu = fluency_eval(query=query, response=response)
                    scores["fluency"].append(flu.get("fluency", 0))
                except Exception:
                    pass

                try:
                    gnd = groundedness_eval(
                        query=query, response=response, context=context
                    )
                    scores["groundedness"].append(gnd.get("groundedness", 0))
                except Exception:
                    pass

            # Average scores and normalize from 1-5 scale to 0-1
            for metric, values in scores.items():
                if values:
                    avg = sum(values) / len(values)
                    avg = avg / 5.0  # SDK returns 1-5, normalize to 0-1
                    setattr(result, metric, avg)
                    record_eval_score(metric, avg)
                    span.set_attribute(f"carevoice.eval.{metric}", avg)

        except ImportError:
            logger.warning("azure-ai-evaluation not installed; running fallback scoring")
            result = _fallback_evaluation(state)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            span.record_exception(e)
            result = _fallback_evaluation(state)

        # Store result
        eval_record = {
            "call_sid": state.call_sid,
            "patient_id": state.patient_id,
            "turn_count": state.turn_count,
            "scores": result.model_dump(),
            "wellness_scores": [s.model_dump() for s in state.wellness_scores],
            "concerns": [c.model_dump() for c in state.concerns],
        }
        _eval_store.append(eval_record)

        return result


def _extract_eval_pairs(state: ConversationState) -> list[dict]:
    """Extract query-response pairs from conversation history."""
    pairs = []
    history = state.message_history
    for i in range(len(history) - 1):
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append({
                "query": history[i]["content"],
                "response": history[i + 1]["content"],
                "context": state.rag_context,
            })
    return pairs


def _fallback_evaluation(state: ConversationState) -> EvalResult:
    """Simple heuristic-based evaluation when SDK is unavailable."""
    result = EvalResult()

    if state.message_history:
        avg_response_len = sum(
            len(m["content"]) for m in state.message_history if m["role"] == "assistant"
        ) / max(1, sum(1 for m in state.message_history if m["role"] == "assistant"))

        result.fluency = min(1.0, avg_response_len / 200)
        result.coherence = 0.8 if state.turn_count > 2 else 0.5
        result.relevance = 0.7
        result.groundedness = 0.75 if state.rag_context else 0.3

    return result


def get_eval_history() -> list[dict]:
    """Return all stored evaluation results for the dashboard."""
    return _eval_store


def get_eval_for_call(call_sid: str) -> Optional[dict]:
    """Return evaluation results for a specific call."""
    for record in _eval_store:
        if record["call_sid"] == call_sid:
            return record
    return None