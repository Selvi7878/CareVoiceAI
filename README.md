# CareVoice AI — Governed Voice + RAG System

CareVoice AI is a voice-enabled AI system designed for regulated environments such as healthcare and compliance. It delivers **grounded, safe, and observable conversations** using Retrieval-Augmented Generation (RAG) and Azure-native services.

---

## Overview

CareVoice AI enables real-time wellness conversations with patients through voice while ensuring:

- Responses are grounded in data (RAG)
- Safety checks are enforced on every turn
- System behavior is fully observable
- Outputs are evaluated for quality

The system combines voice interaction, retrieval pipelines, and evaluation frameworks to produce **trustworthy conversational AI**.

---

## Architecture

![CareVoice AI Architecture](src/backend/doc/carevoiceai-architecture-diagram.png)

> End-to-end system architecture showing voice ingestion, RAG pipeline, safety layer, evaluation, and observability.

---

## System Flow (Actual Implementation)

1. User initiates a call via Twilio
2. Audio is streamed to FastAPI via webhook/WebSocket
3. Patient speech is processed into text
4. System checks for **end-of-call intent (bye / thank you)**
5. Query enters RAG pipeline:
   - Embedding generation (Azure OpenAI)
   - Hybrid retrieval (Azure AI Search)
6. Retrieved context is passed to LLM
7. LLM generates response
8. Azure Content Safety validates output
9. Response returned via Twilio voice
10. OpenTelemetry logs traces and metrics
11. Post-call evaluation computes LLM quality metrics

---

## Technology Stack (Live System)

### Backend

- FastAPI
- Python (async + httpx)

### AI & RAG

- Azure OpenAI (GPT-4o-mini)
- Azure OpenAI Embeddings
- Azure AI Search (hybrid retrieval)

### Voice

- Twilio Voice API
- ConversationRelay (streaming)

### Safety

- Azure AI Content Safety

### Data & Storage

- Azure Cosmos DB (patient + session data)

### Observability

- OpenTelemetry (OTLP exporter)
- Azure Application Insights
- Aspire Dashboard

### Evaluation

- Azure AI Evaluation (`azure-ai-evaluation`)
  - groundedness
  - relevance
  - coherence
  - fluency

### Frontend Dashboard

- React
- Vite
- TypeScript
- Recharts

---

## Dashboard

### Wellness Monitoring

![Wellness Dashboard](docs/screenshots/wellness.png)

Tracks:

- Nutrition
- Physical
- Emotional
- Social

Includes radar visualization + live transcript

### RAG Governance & Evaluation

![RAG Governance](docs/screenshots/rag.png)

- Groundedness, relevance, coherence, fluency
- Citation tracking
- Safety checks

### Pipeline View

![Pipeline](docs/screenshots/pipeline.png)

- Triage → Retrieval → Response → Safety → Alert
- Currently implemented procedurally

### Observability

![Observability](docs/screenshots/observability.png)

- OpenTelemetry traces
- Performance metrics
- Safety + evaluation spans

---

## Key Capabilities

### Retrieval-Augmented Generation (RAG)

- Hybrid retrieval using Azure AI Search
- Responses grounded in data

### Safety Enforcement

- Azure Content Safety on every turn
- Guardrails:
  - Medical advice restriction
  - PHI awareness
  - Groundedness checks

### Observability

- Tracked spans:
  - `carevoice.start_call`
  - `carevoice.handle_utterance`
  - `content_safety_check`
  - `cosmos_lookup`
  - `evaluation_run`

### LLM Evaluation

- Groundedness
- Relevance
- Coherence
- Fluency

### Intelligent Call Termination

Detects:

- "bye"
- "thank you"
- "that's all"
- fuzzy speech like "buh"

Triggers Twilio hang-up.

---

## Project Structure

```
backend/
├── main.py
├── api/
├── workflows/
├── rag/
├── agents/
├── utils/
├── eval/
├── otel/
└── doc/

dashboard/
├── src/
│   ├── components/
│   └── charts/

docs/
└── screenshots/
```

---

## Running Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

```bash
# Dashboard
cd dashboard
npm install
npm run dev
```

---

## Design Principles

**Grounded Responses** — Outputs are backed by retrieved data.

**Safety by Default** — All responses pass safety checks.

**Observability First** — Everything is traceable.

**Deterministic Control** — Explicit logic ensures reliability.

---

## Future Enhancements

- Full Microsoft Agent Framework orchestration
- Multi-agent routing
- Personalization memory
- Clinician escalation
- Improved evaluation pipeline

---

## Summary

CareVoice AI demonstrates how voice interfaces, RAG pipelines, safety systems, and observability can be combined to build trustworthy and production-ready AI systems for regulated environments.
