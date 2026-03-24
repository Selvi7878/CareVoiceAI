import { useState, useEffect, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Legend,
} from "recharts";
import { api } from "./api/client";
import type { Session, EvalRecord, AgentPipeline, Citation, CitationSummary, SafetySummary } from "./types";

const POLL_MS = 3000;

const SEV: Record<string, { color: string; bg: string }> = {
  critical: { color: "#DC2626", bg: "#FEF2F2" },
  high: { color: "#EA580C", bg: "#FFF7ED" },
  medium: { color: "#D97706", bg: "#FFFBEB" },
  low: { color: "#059669", bg: "#ECFDF5" },
};

const DIM: Record<string, { icon: string; label: string }> = {
  physical: { icon: "\uD83D\uDCAA", label: "Physical" },
  emotional: { icon: "\uD83D\uDE0A", label: "Emotional" },
  cognitive: { icon: "\uD83E\uDDE0", label: "Cognitive" },
  nutrition: { icon: "\uD83C\uDF4E", label: "Nutrition" },
  social: { icon: "\uD83D\uDC65", label: "Social" },
};

const PHASE: Record<string, string> = {
  greeting: "\uD83D\uDC4B Greeting", identity: "\uD83D\uDD10 Identity", physical: "\uD83D\uDCAA Physical", emotional: "\uD83D\uDE0A Emotional",
  cognitive: "\uD83E\uDDE0 Cognitive", nutrition: "\uD83C\uDF4E Nutrition", social: "\uD83D\uDC65 Social", closing: "\u2705 Closing",
};

const ELDER_COLORS = ["#7C3AED", "#2563EB", "#059669", "#DC2626", "#D97706", "#EC4899", "#0891B2", "#4F46E5"];
const elderColor = (id: string) => ELDER_COLORS[Math.abs([...id].reduce((a, c) => a + c.charCodeAt(0), 0)) % ELDER_COLORS.length];
const elderInitials = (id: string) => {
  const parts = id.replace(/[-_]/g, " ").split(" ").filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return id.slice(0, 2).toUpperCase();
};
const scoreColor = (s: number) => s >= 7 ? "#059669" : s >= 4 ? "#D97706" : "#DC2626";
const statusOf = (scores: { score: number }[]) => {
  if (!scores.length) return { label: "In Progress", color: "#2563EB", bg: "#EFF6FF" };
  const min = Math.min(...scores.map(s => s.score));
  if (min <= 3) return { label: "Needs Attention", color: "#DC2626", bg: "#FEF2F2" };
  if (min <= 5) return { label: "Declining", color: "#D97706", bg: "#FFFBEB" };
  return { label: "Stable", color: "#059669", bg: "#ECFDF5" };
};

export default function App() {
  const [tab, setTab] = useState<"monitor" | "rag" | "agents" | "otel">("monitor");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [evals, setEvals] = useState<EvalRecord[]>([]);
  const [pipeline, setPipeline] = useState<AgentPipeline | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Session | null>(null);
  const [online, setOnline] = useState(false);
  const [lastUpdate, setLastUpdate] = useState("");
  const [allCitations, setAllCitations] = useState<Citation[]>([]);
  const [citationSummary, setCitationSummary] = useState<CitationSummary | null>(null);
  const [safetySummary, setSafetySummary] = useState<SafetySummary | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, e, p, cs, ss] = await Promise.all([
        api.sessions(), api.evals(), api.agents(), api.citationSummary(), api.safetySummary()
      ]);
      setSessions(s); setEvals(e); setPipeline(p); setCitationSummary(cs); setSafetySummary(ss);
      setOnline(true); setLastUpdate(new Date().toLocaleTimeString());
    } catch { setOnline(false); }
  }, []);

  useEffect(() => { refresh(); const id = setInterval(refresh, POLL_MS); return () => clearInterval(id); }, [refresh]);
  useEffect(() => {
    if (!selected) { setDetail(null); setAllCitations([]); return; }
    api.session(selected).then(setDetail).catch(() => setDetail(null));
    api.citationsForCall(selected).then(setAllCitations).catch(() => setAllCitations([]));
  }, [selected, sessions]);

  const tabs = [
    { key: "monitor" as const, label: "Wellness Monitor", icon: "\u2764\uFE0F" },
    { key: "rag" as const, label: "RAG Governance", icon: "\uD83D\uDEE1\uFE0F" },
    { key: "agents" as const, label: "Agent Pipeline", icon: "\uD83E\uDD16" },
    { key: "otel" as const, label: "Observability", icon: "\uD83D\uDCCA" },
  ];

  return (
    <div style={{ fontFamily: "'Nunito', 'Segoe UI', system-ui, sans-serif", background: "#F8FAFC", color: "#1E293B", minHeight: "100vh" }}>
      <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet" />

      <header style={{ background: "linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #2563EB 100%)", color: "#fff", padding: "16px 28px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ width: 42, height: 42, borderRadius: 12, background: "rgba(255,255,255,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22 }}>
              {"\uD83D\uDC9C"}
            </div>
            <div>
              <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>CareVoice AI</div>
              <div style={{ fontSize: 12, opacity: 0.8 }}>Elder Wellness Companion {"\u2022"} Powered by Microsoft Agent Framework</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(255,255,255,0.15)", padding: "6px 12px", borderRadius: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: online ? "#4ADE80" : "#F87171", boxShadow: online ? "0 0 8px #4ADE80" : "none" }} />
              <span style={{ fontSize: 13, fontWeight: 600 }}>{online ? "Live" : "Offline"}</span>
              {online && <span style={{ fontSize: 11, opacity: 0.7 }}>{"\u2022"} {lastUpdate}</span>}
            </div>
            <div style={{ background: "rgba(255,255,255,0.15)", padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
              MAF v{pipeline?.version || "..."}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 4, marginTop: 14 }}>
          {tabs.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)} style={{
              padding: "8px 18px", borderRadius: 8, border: "none", cursor: "pointer",
              fontSize: 14, fontWeight: 700, fontFamily: "inherit",
              background: tab === t.key ? "#fff" : "rgba(255,255,255,0.12)",
              color: tab === t.key ? "#4F46E5" : "rgba(255,255,255,0.85)",
              transition: "all 0.2s",
            }}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </header>

      <div style={{ display: "flex", minHeight: "calc(100vh - 130px)" }}>
        <aside style={{ width: 260, minWidth: 260, background: "#fff", borderRight: "1px solid #E2E8F0", padding: 16, overflowY: "auto" }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: "#7C3AED", letterSpacing: 0.5, marginBottom: 12 }}>
            {"\uD83D\uDC65"} Patients ({sessions.length})
          </div>
          {sessions.length === 0 && (
            <div style={{ background: "#F1F5F9", borderRadius: 12, padding: 20, textAlign: "center" }}>
              <div style={{ fontSize: 36, marginBottom: 8 }}>{"\uD83D\uDCDE"}</div>
              <div style={{ fontSize: 14, color: "#64748B", lineHeight: 1.5 }}>No calls yet.<br/>Call your Twilio number to start a wellness check.</div>
            </div>
          )}
          {sessions.map(s => {
            const st = statusOf(s.wellness_scores);
            const active = selected === s.call_sid;
            const clr = elderColor(s.patient_id);
            return (
              <div key={s.call_sid} onClick={() => setSelected(s.call_sid)} style={{
                padding: 12, borderRadius: 12, marginBottom: 8, cursor: "pointer",
                background: active ? "#F5F3FF" : "#fff",
                border: active ? "2px solid #7C3AED" : "1px solid #E2E8F0",
                transition: "all 0.15s",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: "50%", flexShrink: 0,
                    background: `${clr}15`, border: `2px solid ${clr}40`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 16, fontWeight: 800, color: clr,
                  }}>
                    {elderInitials(s.patient_id)}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 15, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {s.patient_id}
                    </div>
                    <div style={{ fontSize: 12, color: "#64748B", marginTop: 2 }}>
                      {PHASE[s.phase] || s.phase} {"\u2022"} {s.turn_count} turns
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 20, background: st.bg, color: st.color }}>{st.label}</span>
                  {s.concerns.some(c => c.severity === "critical") && (
                    <span style={{ fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 20, background: "#FEF2F2", color: "#DC2626" }}>{"\uD83D\uDEA8"} Alert</span>
                  )}
                </div>
              </div>
            );
          })}
        </aside>

        <main style={{ flex: 1, overflowY: "auto", padding: 24 }}>
          {tab === "monitor" && <MonitorTab detail={detail} selected={selected} citations={allCitations} />}
          {tab === "rag" && <RAGTab evals={evals} citationSummary={citationSummary} safetySummary={safetySummary} allCitations={allCitations} selectedCall={selected} />}
          {tab === "agents" && <AgentsTab pipeline={pipeline} />}
          {tab === "otel" && <OTelTab pipeline={pipeline} sessions={sessions} />}
        </main>
      </div>
    </div>
  );
}

/* ─── Shared ───────────────────────────────────────────────── */

function Card({ title, icon, children, accent, style }: {
  title?: string; icon?: string; children: React.ReactNode; accent?: string; style?: React.CSSProperties;
}) {
  return (
    <div style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", padding: 20, boxShadow: "0 1px 3px rgba(0,0,0,0.04)", ...style }}>
      {title && (
        <div style={{ fontSize: 14, fontWeight: 800, color: accent || "#7C3AED", marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
          {icon && <span style={{ fontSize: 18 }}>{icon}</span>}
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function Empty({ icon, msg }: { icon: string; msg: string }) {
  return (
    <div style={{ textAlign: "center", padding: 60 }}>
      <div style={{ fontSize: 48, marginBottom: 12 }}>{icon}</div>
      <div style={{ fontSize: 16, color: "#94A3B8", lineHeight: 1.5 }}>{msg}</div>
    </div>
  );
}

/* ═══ TAB 1: WELLNESS MONITOR ══════════════════════════════ */

function MonitorTab({ detail, selected, citations }: { detail: Session | null; selected: string | null; citations: Citation[] }) {
  if (!selected || !detail) return <Empty icon={"\uD83D\uDC68\u200D\u2695\uFE0F"} msg="Select a patient from the sidebar to view their wellness report" />;

  const radar = detail.wellness_scores.map(s => ({ dimension: DIM[s.dimension]?.label || s.dimension, score: s.score, fullMark: 10 }));
  const clr = elderColor(detail.patient_id);

  // Build a map of turn -> citation for conversation display
  const citationByTurn: Record<number, Citation> = {};
  citations.forEach(c => { citationByTurn[c.turn] = c; });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {/* Patient header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", padding: "16px 20px" }}>
        <div style={{
          width: 56, height: 56, borderRadius: "50%", background: `${clr}15`, border: `3px solid ${clr}`,
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, fontWeight: 800, color: clr,
        }}>
          {elderInitials(detail.patient_id)}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 20, fontWeight: 800 }}>{detail.patient_id}</div>
          <div style={{ fontSize: 14, color: "#64748B", marginTop: 2 }}>
            {PHASE[detail.phase] || detail.phase} {"\u2022"} {detail.turn_count} conversation turns {"\u2022"} {detail.call_type.replace("_", " ")}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {(() => { const st = statusOf(detail.wellness_scores); return <span style={{ fontSize: 14, fontWeight: 700, padding: "6px 16px", borderRadius: 20, background: st.bg, color: st.color }}>{st.label}</span>; })()}
        </div>
      </div>

      {/* Score cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
        {detail.wellness_scores.map(s => {
          const d = DIM[s.dimension];
          return (
            <div key={s.dimension} style={{
              background: "#fff", borderRadius: 14, border: "1px solid #E2E8F0", padding: 16, textAlign: "center",
              boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
            }}>
              <div style={{ fontSize: 28, marginBottom: 4 }}>{d?.icon || "\u2753"}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#64748B" }}>{d?.label || s.dimension}</div>
              <div style={{ fontSize: 36, fontWeight: 800, color: scoreColor(s.score), marginTop: 4 }}>
                {s.score}<span style={{ fontSize: 16, color: "#CBD5E1" }}>/10</span>
              </div>
              <div style={{ fontSize: 11, color: "#94A3B8", marginTop: 6, lineHeight: 1.4 }}>{s.reasoning}</div>
            </div>
          );
        })}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {radar.length > 0 && (
          <Card title="Wellness Overview" icon={"\uD83D\uDCC8"} accent="#059669">
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radar}>
                <PolarGrid stroke="#E2E8F0" />
                <PolarAngleAxis dataKey="dimension" tick={{ fill: "#64748B", fontSize: 13, fontWeight: 600 }} />
                <PolarRadiusAxis domain={[0, 10]} tick={{ fill: "#94A3B8", fontSize: 11 }} />
                <Radar dataKey="score" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.2} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        )}

        <Card title={`Concerns Flagged (${detail.concerns.length})`} icon={"\u26A0\uFE0F"} accent="#DC2626">
          {detail.concerns.length === 0
            ? <div style={{ fontSize: 14, color: "#94A3B8", padding: 20, textAlign: "center" }}>{"\u2705"} No concerns detected during this call</div>
            : detail.concerns.map((c, i) => {
              const sv = SEV[c.severity] || SEV.low;
              return (
                <div key={i} style={{ padding: 12, borderRadius: 12, marginBottom: 8, background: sv.bg, borderLeft: `4px solid ${sv.color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 14, fontWeight: 700, color: "#1E293B" }}>{c.category}</span>
                    <span style={{ fontSize: 12, fontWeight: 700, padding: "2px 10px", borderRadius: 12, background: sv.color, color: "#fff" }}>{c.severity}</span>
                  </div>
                  <div style={{ fontSize: 13, color: "#475569", marginTop: 6 }}>{c.description}</div>
                  <div style={{ fontSize: 12, color: "#7C3AED", marginTop: 4, fontWeight: 600 }}>{"\u27A1\uFE0F"} {c.suggested_action}</div>
                </div>
              );
            })}
        </Card>
      </div>

      {/* Conversation with inline citations */}
      <Card title="Live Conversation" icon={"\uD83D\uDCAC"} accent="#2563EB">
        {!detail.message_history?.length
          ? <div style={{ fontSize: 14, color: "#94A3B8", textAlign: "center", padding: 24 }}>Conversation will appear here as CareVoice AI talks with the patient...</div>
          : <div style={{ maxHeight: 420, overflowY: "auto" }}>
              {detail.message_history.map((m, i) => {
                const isPatient = m.role === "user";
                // Find citation for this assistant turn
                const turnIndex = detail.message_history!.slice(0, i + 1).filter(x => x.role === "assistant").length;
                const citation = !isPatient ? citationByTurn[turnIndex] : null;
                return (
                  <div key={i} style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                    <div style={{
                      width: 36, height: 36, borderRadius: "50%", flexShrink: 0,
                      background: isPatient ? "#ECFDF5" : "#F5F3FF",
                      border: `2px solid ${isPatient ? "#059669" : "#7C3AED"}`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 14, fontWeight: 700, color: isPatient ? "#059669" : "#7C3AED",
                    }}>
                      {isPatient ? elderInitials(detail.patient_id) : "\uD83C\uDF3C"}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        background: isPatient ? "#F0FDF4" : "#FAF5FF",
                        borderRadius: 12, padding: "10px 14px",
                        border: `1px solid ${isPatient ? "#BBF7D0" : "#E9D5FF"}`,
                      }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: isPatient ? "#059669" : "#7C3AED", marginBottom: 4 }}>
                          {isPatient ? "Patient" : "CareVoice AI"}
                        </div>
                        <div style={{ fontSize: 14, color: "#334155", lineHeight: 1.6 }}>{m.content}</div>
                      </div>
                      {/* Citation badges under assistant messages */}
                      {citation && citation.sources_cited.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4, paddingLeft: 4 }}>
                          {citation.sources_cited.map((src, si) => (
                            <span key={si} style={{
                              fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 8,
                              background: "#EFF6FF", color: "#2563EB", border: "1px solid #BFDBFE",
                            }}>
                              {"\uD83D\uDD17"} {src.field.replace("patient.", "")}
                            </span>
                          ))}
                          <span style={{
                            fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 8,
                            background: citation.groundedness_score >= 0.8 ? "#ECFDF5" : "#FEF2F2",
                            color: citation.groundedness_score >= 0.8 ? "#059669" : "#DC2626",
                          }}>
                            {Math.round(citation.groundedness_score * 100)}% grounded
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>}
      </Card>
    </div>
  );
}

/* ═══ TAB 2: RAG GOVERNANCE ════════════════════════════════ */

function RAGTab({ evals, citationSummary, safetySummary, allCitations, selectedCall }: {
  evals: EvalRecord[]; citationSummary: CitationSummary | null; safetySummary: SafetySummary | null;
  allCitations: Citation[]; selectedCall: string | null;
}) {
  const latest = evals.length > 0 ? evals[evals.length - 1].scores : null;
  const trend = evals.map((e, i) => ({
    call: `Call ${i + 1}`,
    Groundedness: Math.round(e.scores.groundedness * 100),
    Relevance: Math.round(e.scores.relevance * 100),
    Coherence: Math.round(e.scores.coherence * 100),
    Fluency: Math.round(e.scores.fluency * 100),
  }));

  const metrics = [
    { key: "groundedness", label: "Groundedness", desc: "Responses backed by patient records", icon: "\uD83C\uDFAF" },
    { key: "relevance", label: "Relevance", desc: "Answers match patient's question", icon: "\u2705" },
    { key: "coherence", label: "Coherence", desc: "Logical and consistent responses", icon: "\uD83E\uDDE9" },
    { key: "fluency", label: "Fluency", desc: "Natural, well-formed language", icon: "\u270D\uFE0F" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

      {/* Citation summary stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {[
          { label: "Total Responses", value: citationSummary?.total_responses ?? 0, icon: "\uD83D\uDCAC", color: "#4F46E5" },
          { label: "Cited Sources", value: citationSummary?.total_source_references ?? 0, icon: "\uD83D\uDD17", color: "#2563EB" },
          { label: "Avg Groundedness", value: `${Math.round((citationSummary?.avg_groundedness ?? 0) * 100)}%`, icon: "\uD83C\uDFAF", color: (citationSummary?.avg_groundedness ?? 0) >= 0.8 ? "#059669" : "#D97706" },
          { label: "Safety Checks", value: `${safetySummary?.passed ?? 0}/${safetySummary?.total_checks ?? 0} passed`, icon: "\uD83D\uDEE1\uFE0F", color: (safetySummary?.flagged ?? 0) === 0 ? "#059669" : "#DC2626" },
        ].map(s => (
          <div key={s.label} style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", padding: 18, textAlign: "center", boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
            <div style={{ fontSize: 28, marginBottom: 4 }}>{s.icon}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#64748B" }}>{s.label}</div>
            <div style={{ fontSize: 28, fontWeight: 800, color: s.color, marginTop: 4 }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Eval score cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {metrics.map(m => {
          const v = latest ? (latest as any)[m.key] as number : 0;
          const p = Math.round(v * 100);
          const c = v >= 0.8 ? "#059669" : v >= 0.6 ? "#D97706" : "#DC2626";
          const bg = v >= 0.8 ? "#ECFDF5" : v >= 0.6 ? "#FFFBEB" : "#FEF2F2";
          return (
            <div key={m.key} style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", padding: 20, textAlign: "center", boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
              <div style={{ fontSize: 28, marginBottom: 6 }}>{m.icon}</div>
              <div style={{ fontSize: 14, fontWeight: 800, color: "#64748B" }}>{m.label}</div>
              <div style={{ fontSize: 42, fontWeight: 800, color: c, margin: "8px 0" }}>{p}%</div>
              <div style={{ fontSize: 12, color: "#94A3B8" }}>{m.desc}</div>
              <div style={{ marginTop: 8, display: "inline-block", fontSize: 12, fontWeight: 700, padding: "3px 12px", borderRadius: 12, background: bg, color: c }}>
                {v >= 0.8 ? "Excellent" : v >= 0.6 ? "Needs Work" : "Below Threshold"}
              </div>
            </div>
          );
        })}
      </div>

      {trend.length > 0 && (
        <Card title="Evaluation Trends" icon={"\uD83D\uDCC8"} accent="#4F46E5">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={trend}>
              <CartesianGrid stroke="#F1F5F9" />
              <XAxis dataKey="call" tick={{ fill: "#94A3B8", fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fill: "#94A3B8", fontSize: 12 }} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E8F0", fontSize: 13, fontWeight: 600 }} />
              <Area type="monotone" dataKey="Groundedness" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.1} strokeWidth={2} />
              <Area type="monotone" dataKey="Relevance" stroke="#059669" fill="#059669" fillOpacity={0.1} strokeWidth={2} />
              <Area type="monotone" dataKey="Coherence" stroke="#D97706" fill="#D97706" fillOpacity={0.1} strokeWidth={2} />
              <Area type="monotone" dataKey="Fluency" stroke="#2563EB" fill="#2563EB" fillOpacity={0.1} strokeWidth={2} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 13, fontWeight: 600 }} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Live citation trail for selected call */}
      <Card title={`Citation Audit Trail${allCitations.length > 0 ? ` (${allCitations.length} responses tracked)` : ""}`} icon={"\uD83D\uDD17"} accent="#2563EB">
        {allCitations.length === 0
          ? <div style={{ fontSize: 14, color: "#94A3B8", textAlign: "center", padding: 20 }}>Select a call from the sidebar to see per-response citations</div>
          : <div style={{ maxHeight: 300, overflowY: "auto" }}>
              {allCitations.map((c, i) => (
                <div key={i} style={{ padding: "10px 0", borderBottom: "1px solid #F1F5F9" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color: "#7C3AED", background: "#F5F3FF", padding: "2px 8px", borderRadius: 6 }}>Turn {c.turn}</span>
                    <span style={{
                      fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 8,
                      background: c.groundedness_score >= 0.8 ? "#ECFDF5" : "#FEF2F2",
                      color: c.groundedness_score >= 0.8 ? "#059669" : "#DC2626",
                    }}>
                      {Math.round(c.groundedness_score * 100)}% grounded
                    </span>
                    {c.sources_cited.length > 0 && (
                      <span style={{ fontSize: 11, color: "#2563EB", fontWeight: 600 }}>{c.sources_cited.length} source{c.sources_cited.length > 1 ? "s" : ""} cited</span>
                    )}
                    {c.ungrounded_claims.length > 0 && (
                      <span style={{ fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 8, background: "#FEF2F2", color: "#DC2626" }}>{"\u26A0\uFE0F"} Hallucination flag</span>
                    )}
                  </div>
                  <div style={{ fontSize: 13, color: "#475569", lineHeight: 1.4 }}>{c.response.slice(0, 120)}{c.response.length > 120 ? "..." : ""}</div>
                  {c.sources_cited.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                      {c.sources_cited.map((src, si) => (
                        <span key={si} style={{ fontSize: 10, padding: "2px 6px", borderRadius: 6, background: "#EFF6FF", color: "#2563EB", border: "1px solid #BFDBFE" }}>
                          {src.fragment}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
        }
      </Card>

      {/* Safety layers — updated to match what's actually running */}
      <Card title="Responsible AI Safety Layers" icon={"\uD83D\uDEE1\uFE0F"} accent="#DC2626">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
          {[
            { n: "Azure Content Safety", d: "Screens for hate, self-harm, sexual, violence", icon: "\uD83D\uDEAB", live: true },
            { n: "Medical Guardrails", d: "Prevents diagnoses and prescription changes", icon: "\uD83C\uDFE5", live: true },
            { n: "Groundedness Check", d: "Verifies claims against patient records", icon: "\uD83D\uDCCB", live: true },
            { n: "Elder Respect", d: "Catches condescending or ageist language", icon: "\u2696\uFE0F", live: true },
            { n: "PHI Protection", d: "No patient data before identity verification", icon: "\uD83D\uDD10", live: true },
            { n: "Citation Tracking", d: "Every response traced to source documents", icon: "\uD83D\uDD17", live: true },
          ].map(s => (
            <div key={s.n} style={{ background: "#F8FAFC", borderRadius: 12, padding: 14, border: "1px solid #E2E8F0" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 20 }}>{s.icon}</span>
                <span style={{ fontSize: 14, fontWeight: 700, color: "#1E293B" }}>{s.n}</span>
                <div style={{ marginLeft: "auto", width: 8, height: 8, borderRadius: "50%", background: s.live ? "#4ADE80" : "#D4D4D8" }} />
              </div>
              <div style={{ fontSize: 13, color: "#64748B", lineHeight: 1.4 }}>{s.d}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Eval audit trail */}
      <Card title="Evaluation Audit Trail" icon={"\uD83D\uDCDD"} accent="#EA580C">
        {evals.length === 0 ? <div style={{ fontSize: 14, color: "#94A3B8", textAlign: "center", padding: 20 }}>Complete a call to see the audit trail</div> : (
          <div>
            {evals.map((e, i) => (
              <div key={i} style={{ padding: "12px 0", borderBottom: "1px solid #F1F5F9", display: "flex", alignItems: "center", gap: 12 }}>
                <code style={{ fontSize: 13, color: "#7C3AED", fontWeight: 700, background: "#F5F3FF", padding: "2px 8px", borderRadius: 6 }}>...{e.call_sid.slice(-6)}</code>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{e.patient_id}</span>
                <span style={{ fontSize: 12, color: "#94A3B8" }}>{e.turn_count} turns</span>
                <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                  {(["groundedness", "relevance", "coherence", "fluency"] as const).map(m => {
                    const v = (e.scores as any)[m] as number;
                    const c = v >= 0.8 ? "#059669" : v >= 0.6 ? "#D97706" : "#DC2626";
                    return <span key={m} style={{ fontSize: 12, fontWeight: 700, padding: "3px 10px", borderRadius: 12, background: v >= 0.8 ? "#ECFDF5" : v >= 0.6 ? "#FFFBEB" : "#FEF2F2", color: c }}>{m.slice(0, 4)} {Math.round(v * 100)}%</span>;
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

/* ═══ TAB 3: AGENTS ════════════════════════════════════════ */

function AgentsTab({ pipeline }: { pipeline: AgentPipeline | null }) {
  if (!pipeline) return <Empty icon={"\uD83E\uDD16"} msg="Loading agent pipeline..." />;
  const agentInfo = [
    { color: "#059669", bg: "#ECFDF5", icon: "\uD83D\uDCDE" },
    { color: "#2563EB", bg: "#EFF6FF", icon: "\uD83D\uDD0D" },
    { color: "#EA580C", bg: "#FFF7ED", icon: "\uD83D\uDCAC" },
    { color: "#DC2626", bg: "#FEF2F2", icon: "\uD83D\uDEE1\uFE0F" },
    { color: "#D97706", bg: "#FFFBEB", icon: "\uD83D\uDEA8" },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <Card title="How CareVoice AI Works" icon={"\uD83E\uDD16"} accent="#4F46E5">
        <div style={{ fontSize: 14, color: "#64748B", marginBottom: 16, lineHeight: 1.6 }}>
          Five specialized AI components work together in a governed pipeline. Each handles one responsibility and passes results to the next.
        </div>
        <div style={{ display: "flex", alignItems: "stretch", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
          {pipeline.agents.map((a, i) => {
            const info = agentInfo[i];
            return (
              <div key={a.name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  background: info.bg, border: `2px solid ${info.color}40`, borderRadius: 14,
                  padding: 16, textAlign: "center", minWidth: 130,
                }}>
                  <div style={{ fontSize: 28, marginBottom: 6 }}>{info.icon}</div>
                  <div style={{ fontSize: 14, fontWeight: 800, color: info.color }}>{a.name}</div>
                  <div style={{ fontSize: 12, color: "#64748B", marginTop: 4 }}>{a.role}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "center", marginTop: 8 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: a.status === "active" ? "#4ADE80" : "#F87171" }} />
                    <span style={{ fontSize: 12, fontWeight: 700, color: a.status === "active" ? "#059669" : "#DC2626" }}>{a.status}</span>
                  </div>
                </div>
                {i < pipeline.agents.length - 1 && <div style={{ fontSize: 22, color: "#CBD5E1", fontWeight: 800 }}>{"\u279C"}</div>}
              </div>
            );
          })}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card title="Azure AI Stack" icon={"\u2601\uFE0F"} accent="#2563EB">
          {[
            { s: "Azure OpenAI (GPT-4o-mini)", live: true },
            { s: "Azure Cosmos DB", live: true },
            { s: "Azure Content Safety", live: true },
            { s: "Azure AI Evaluation SDK", live: true },
            { s: "Azure App Insights", live: true },
            { s: "Azure AI Search", live: false },
            { s: "Azure Speech", live: false },
          ].map(x => (
            <div key={x.s} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "1px solid #F1F5F9" }}>
              <span style={{ fontSize: 14, color: "#334155" }}>{x.s}</span>
              <span style={{ fontSize: 12, fontWeight: 700, color: x.live ? "#059669" : "#94A3B8", background: x.live ? "#ECFDF5" : "#F1F5F9", padding: "2px 10px", borderRadius: 10 }}>
                {x.live ? "\u2713 Live" : "Provisioned"}
              </span>
            </div>
          ))}
        </Card>
        <Card title="Configuration" icon={"\u2699\uFE0F"} accent="#7C3AED">
          <Row2 label="Framework" value={`${pipeline.framework} ${pipeline.version}`} />
          <Row2 label="OpenTelemetry" value={pipeline.observability.otel_enabled ? "Enabled" : "Disabled"} ok={pipeline.observability.otel_enabled} />
          <Row2 label="App Insights" value={pipeline.observability.app_insights ? "Connected" : "Not connected"} ok={pipeline.observability.app_insights} />
          <Row2 label="Eval SDK" value={pipeline.evaluation.sdk} />
          {pipeline.evaluation.metrics.map(m => <Row2 key={m} label={m} value="Active" ok={true} />)}
        </Card>
      </div>
    </div>
  );
}

function Row2({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "1px solid #F1F5F9" }}>
      <span style={{ fontSize: 14, color: "#64748B" }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 700, color: ok !== undefined ? (ok ? "#059669" : "#DC2626") : "#334155" }}>{value}</span>
    </div>
  );
}

/* ═══ TAB 4: OBSERVABILITY ═════════════════════════════════ */

function OTelTab({ pipeline, sessions }: { pipeline: AgentPipeline | null; sessions: Session[] }) {
  const data = sessions.map(s => ({ Call: s.call_sid.slice(-6), Turns: s.turn_count, Concerns: s.concerns.length, "Safety Flags": s.safety_flags.length }));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", padding: 20 }}>
        <div style={{ fontSize: 16, fontWeight: 800, color: "#1E293B", marginBottom: 4 }}>{"\uD83D\uDD0D"} What is Observability?</div>
        <div style={{ fontSize: 14, color: "#64748B", lineHeight: 1.7 }}>
          Every time CareVoice AI talks to a patient, the system records exactly what happened — which component ran, how long it took, what data was retrieved, and whether safety checks passed. This is called "distributed tracing" and it helps us ensure CareVoice AI is working correctly and safely.
        </div>
      </div>

      {data.length > 0 && (
        <Card title="Call Activity Overview" icon={"\uD83D\uDCCA"} accent="#059669">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data}>
              <CartesianGrid stroke="#F1F5F9" />
              <XAxis dataKey="Call" tick={{ fill: "#94A3B8", fontSize: 12 }} />
              <YAxis tick={{ fill: "#94A3B8", fontSize: 12 }} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E8F0", fontSize: 13 }} />
              <Bar dataKey="Turns" fill="#7C3AED" radius={[6, 6, 0, 0]} />
              <Bar dataKey="Concerns" fill="#D97706" radius={[6, 6, 0, 0]} />
              <Bar dataKey="Safety Flags" fill="#DC2626" radius={[6, 6, 0, 0]} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 13, fontWeight: 600 }} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card title="Trace Spans Emitted" icon={"\uD83D\uDD17"} accent="#7C3AED">
          {[
            { s: "carevoice.start_call", d: "Call initialization + patient lookup" },
            { s: "carevoice.safety_check", d: "5-layer safety middleware" },
            { s: "carevoice.evaluate_conversation", d: "Post-call quality grading" },
          ].map(x => (
            <div key={x.s} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #F1F5F9", alignItems: "center" }}>
              <code style={{ fontSize: 13, color: "#7C3AED", fontWeight: 700, background: "#F5F3FF", padding: "2px 8px", borderRadius: 6 }}>{x.s}</code>
              <span style={{ fontSize: 13, color: "#64748B" }}>{x.d}</span>
            </div>
          ))}
        </Card>
        <Card title="Performance Metrics" icon={"\u26A1"} accent="#D97706">
          {[
            { m: "carevoice.calls.started", t: "Counter", d: "How many calls have been made" },
            { m: "carevoice.calls.duration_seconds", t: "Histogram", d: "How long each call lasted" },
            { m: "carevoice.wellness.score", t: "Histogram", d: "Distribution of wellness scores" },
            { m: "carevoice.safety.checks", t: "Counter", d: "Safety checks passed vs. flagged" },
            { m: "carevoice.eval.score", t: "Histogram", d: "AI quality score distribution" },
          ].map(x => (
            <div key={x.m} style={{ padding: "10px 0", borderBottom: "1px solid #F1F5F9" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <code style={{ fontSize: 13, color: "#D97706", fontWeight: 700 }}>{x.m}</code>
                <span style={{ fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 8, background: "#EFF6FF", color: "#2563EB" }}>{x.t}</span>
              </div>
              <div style={{ fontSize: 13, color: "#64748B", marginTop: 4 }}>{x.d}</div>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}
