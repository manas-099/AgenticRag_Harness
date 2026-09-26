/**
 * ResultCard.tsx — renders the final answer, source chunk chips, and a
 * degraded-run warning when the harness had to fall back mid-verification.
 */
import { useQueryStore } from "@/store/queryStore";

export function ResultCard() {
  const { answer, sourcesUsed, degraded, degradeReason, fromCache, error, mode } = useQueryStore((s) => ({
    answer: s.answer,
    sourcesUsed: s.sourcesUsed,
    degraded: s.degraded,
    degradeReason: s.degradeReason,
    fromCache: s.fromCache,
    error: s.error,
    mode: s.mode,
  }));

  if (error) {
    return <div className="error-banner">{error}</div>;
  }

  if (!answer) {
    return (
      <div className="result-card">
        <div className="result-title">{mode === "standard" ? "Standard RAG" : "Agentic RAG"}</div>
        <div className="empty-hint">
          {mode === "standard"
            ? "Run a query to fetch a direct, single-pass answer."
            : "Run a query to see the agent reason through it step by step."}
        </div>
      </div>
    );
  }

  return (
    <div className="result-card">
      <div className="result-card-head">
        <span className="result-title">{mode === "standard" ? "Standard RAG" : "Agentic RAG"}</span>
        <span className={`result-status ${degraded ? "degraded" : ""}`}>
          {degraded ? `degraded — ${degradeReason ?? "unknown reason"}` : fromCache ? "cache hit" : "verified"}
        </span>
      </div>
      <div className="answer-text">{answer}</div>
      {sourcesUsed.length > 0 && (
        <div className="sources-row">
          {sourcesUsed.map((id) => (
            <span className="source-chip" key={id}>
              {id.slice(0, 8)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
