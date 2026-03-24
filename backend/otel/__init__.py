"""
OpenTelemetry configuration for CareVoice AI.

Sets up tracing and metrics with Azure Monitor exporter
so all spans and metrics flow to Application Insights.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

_initialized = False
_tracer = None
_meter = None


def setup_observability() -> None:
    global _initialized, _tracer, _meter
    if _initialized:
        return

    os.environ.setdefault("ENABLE_INSTRUMENTATION", "true")

    # Try MAF's built-in setup first
    try:
        from agent_framework.observability import configure_otel_providers
        configure_otel_providers()
    except Exception as e:
        logger.warning(f"MAF OTel setup: {e}")

    # Explicitly wire Azure Monitor exporter for App Insights
    conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if conn_str:
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from azure.monitor.opentelemetry.exporter import (
                AzureMonitorTraceExporter,
                AzureMonitorMetricExporter,
            )

            resource = Resource.create({"service.name": "carevoice-ai", "service.version": "2.0.0"})

            # Traces → App Insights
            current_tp = trace.get_tracer_provider()
            if hasattr(current_tp, "add_span_processor"):
                # MAF already set up a TracerProvider, just add our exporter
                trace_exporter = AzureMonitorTraceExporter(connection_string=conn_str)
                current_tp.add_span_processor(BatchSpanProcessor(trace_exporter))
            else:
                # No TracerProvider set yet, create one
                tp = TracerProvider(resource=resource)
                trace_exporter = AzureMonitorTraceExporter(connection_string=conn_str)
                tp.add_span_processor(BatchSpanProcessor(trace_exporter))
                trace.set_tracer_provider(tp)

            # Metrics → App Insights
            metric_exporter = AzureMonitorMetricExporter(connection_string=conn_str)
            reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
            mp = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(mp)

            logger.info("Azure Monitor exporter connected to App Insights")

        except ImportError as e:
            logger.warning(f"Azure Monitor exporter not available: {e}")
        except Exception as e:
            logger.warning(f"Azure Monitor setup failed: {e}")
    else:
        logger.info("No APPLICATIONINSIGHTS_CONNECTION_STRING — skipping Azure Monitor")

    _initialized = True
    logger.info("OTel instrumentation enabled for CareVoice AI")


def get_carevoice_tracer():
    global _tracer
    if _tracer is None:
        try:
            from agent_framework.observability import get_tracer
            _tracer = get_tracer()
        except Exception:
            from opentelemetry import trace
            _tracer = trace.get_tracer("carevoice-ai", "2.0.0")
    return _tracer


def get_carevoice_meter():
    global _meter
    if _meter is None:
        try:
            from agent_framework.observability import get_meter
            _meter = get_meter()
        except Exception:
            from opentelemetry import metrics
            _meter = metrics.get_meter("carevoice-ai", "2.0.0")
    return _meter


# ─── Pre-built metrics ───────────────────────────────────────────────────────

def _ensure_meter():
    return get_carevoice_meter()


def record_call_started(patient_id: str, call_sid: str):
    m = _ensure_meter()
    counter = m.create_counter("carevoice.calls.started")
    counter.add(1, {"patient_id": patient_id, "call_sid": call_sid})


def record_call_ended(patient_id: str, call_sid: str, duration_s: float):
    m = _ensure_meter()
    hist = m.create_histogram("carevoice.calls.duration_seconds")
    hist.record(duration_s, {"patient_id": patient_id, "call_sid": call_sid})


def record_wellness_score(patient_id: str, dimension: str, score: int):
    m = _ensure_meter()
    gauge = m.create_histogram("carevoice.wellness.score")
    gauge.record(score, {"patient_id": patient_id, "dimension": dimension})


def record_safety_check(patient_id: str, is_safe: bool, groundedness: float):
    m = _ensure_meter()
    counter = m.create_counter("carevoice.safety.checks")
    counter.add(1, {
        "patient_id": patient_id,
        "is_safe": str(is_safe),
        "groundedness_bucket": "high" if groundedness >= 0.7 else "low",
    })


def record_eval_score(metric: str, score: float):
    m = _ensure_meter()
    hist = m.create_histogram("carevoice.eval.score")
    hist.record(score, {"metric": metric})