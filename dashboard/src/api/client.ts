import type { Session, EvalRecord, AgentPipeline, Citation, CitationSummary, SafetySummary } from "../types";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get<{ status: string }>("/health"),
  sessions: () => get<Session[]>("/api/sessions"),
  session: (id: string) => get<Session>(`/api/sessions/${id}`),
  evals: () => get<EvalRecord[]>("/api/eval"),
  evalForCall: (id: string) => get<EvalRecord>(`/api/eval/${id}`),
  agents: () => get<AgentPipeline>("/api/agents"),
  citations: () => get<Citation[]>("/api/citations"),
  citationsForCall: (id: string) => get<Citation[]>(`/api/citations/${id}`),
  citationSummary: () => get<CitationSummary>("/api/citations/summary"),
  safetySummary: () => get<SafetySummary>("/api/safety/summary"),
};