export interface WellnessScore {
  dimension: string;
  score: number;
  reasoning: string;
}

export interface Concern {
  category: string;
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  suggested_action: string;
}

export interface Session {
  call_sid: string;
  patient_id: string;
  call_type: string;
  phase: string;
  turn_count: number;
  wellness_scores: WellnessScore[];
  concerns: Concern[];
  safety_flags: string[];
  started_at: string;
  call_ended?: boolean;
  message_history?: { role: string; content: string }[];
}

export interface EvalScores {
  groundedness: number;
  relevance: number;
  coherence: number;
  fluency: number;
  safety_score: number;
  hallucination_flags: string[];
  evaluated_at: string;
}

export interface EvalRecord {
  call_sid: string;
  patient_id: string;
  turn_count: number;
  scores: EvalScores;
  wellness_scores: WellnessScore[];
  concerns: Concern[];
}

export interface AgentInfo {
  name: string;
  role: string;
  status: string;
}

export interface AgentPipeline {
  framework: string;
  version: string;
  agents: AgentInfo[];
  observability: {
    otel_enabled: boolean;
    exporter: string;
    app_insights: boolean;
  };
  evaluation: {
    sdk: string;
    metrics: string[];
  };
}

export interface CitationSource {
  field: string;
  cited_value: string;
  source_document: string;
  fragment: string;
}

export interface Citation {
  call_sid: string;
  turn: number;
  response: string;
  patient_id: string;
  timestamp: string;
  sources_cited: CitationSource[];
  groundedness_score: number;
  ungrounded_claims: string[];
  document_source: string;
}

export interface CitationSummary {
  total_responses: number;
  responses_with_citations: number;
  avg_groundedness: number;
  ungrounded_responses: number;
  total_source_references: number;
}

export interface SafetySummary {
  total_checks: number;
  passed: number;
  flagged: number;
  avg_groundedness: number;
}