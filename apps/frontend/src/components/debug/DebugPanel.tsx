// =============================================================================
// FILE: src/components/debug/DebugPanel.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/debug/DebugPanel.tsx
// NEW FILE
//
// WHAT IT DOES:
//   Floating overlay panel (right side of screen, slides in from right).
//   Toggled by the ⬡ button in the Topbar.
//   Shows a full debug view of the last agentic run:
//
//   Sections:
//   1. QUERY — the question that was asked
//   2. STATUS — degraded Y/N, degrade_reason (human-readable explanation)
//   3. LOOP STATS — iterations_used, validation_retries, token_budget_used
//      with a visual budget bar (used / 12000 max)
//   4. AGENT DECISIONS — for each ActionStep: iteration badge, action chip
//      (color-coded by action type), thought text. This is the "show the
//      decision" feature from the screenshot — no LLM needed, it's already
//      in the response.
//   5. CACHE STATUS — query_cache_size + retrieval_cache_size from health
//      endpoint, shown as "Query cache: N hits" / "Retrieval cache: N hits"
//
//   Empty state: "Run an agentic query to see debug info"
//
// WHY:
//   Notes say "A debug panel also. I want to showcase the Harness the system
//   design so end to end how a query is processed. With a good visualization UI.
//   How many agents running. What agent is what doing, finished or not.
//   Where is the loop. Why degrade."
//   Also: "I also want this kind of decision to show" (screenshot of decision UI)
//
// WHERE IT CONNECTS:
//   store/debugStore.ts (lastResult, isOpen, setOpen)
//   api/healthApi.ts (fetchBackendHealth for cache stats)
// =============================================================================

import { useEffect, useState } from "react";
import { useDebugStore } from "@/store/debugStore";
import { fetchBackendHealth } from "@/api/healthApi";
import type { BackendHealth, ActionStep } from "@/types/domain";

const ACTION_COLORS: Record<string, string> = {
  search_documents: "var(--teal)",
  search_documents_reformulated: "var(--amber)",
  get_page: "var(--teal)",
  compare_chunks: "#a78bfa",
  check_document_freshness: "#60a5fa",
  answer: "var(--teal)",
};

const DEGRADE_EXPLANATIONS: Record<string, string> = {
  max_iterations_exceeded: "The agent hit its maximum iteration limit without finding a confident answer.",
  token_budget_exceeded: "The token budget for this run was exhausted before the agent could finish.",
  max_search_attempts_exceeded: "Too many search attempts were made without finding better results.",
  stuck_loop_no_new_chunks: "The agent detected it was stuck in a loop — repeated searches found no new chunks.",
  max_validation_retries_exceeded: "The generated answer failed verification too many times and was dropped.",
  no_chunks_retrieved: "No relevant chunks were found in the vector store for this query.",
  llm_generation_failed: "The LLM failed to generate an answer — check the LLM chain config.",
};

function ActionStepRow({ step }: { step: ActionStep }) {
  const color = ACTION_COLORS[step.action] ?? "var(--text-muted)";
  return (
    <div className="debug-step-row">
      <div className="debug-step-iter">#{step.iteration}</div>
      <div className="debug-step-content">
        <span className="debug-action-chip" style={{ color, borderColor: color }}>
          {step.action.replace(/_/g, " ")}
        </span>
        {step.thought && (
          <div className="debug-thought">"{step.thought}"</div>
        )}
      </div>
    </div>
  );
}

export function DebugPanel() {
  const { isOpen, setOpen, lastResult } = useDebugStore();
  const [health, setHealth] = useState<BackendHealth | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    fetchBackendHealth().then(setHealth).catch(() => {});
  }, [isOpen]);

  if (!isOpen) return null;

  const MAX_TOKEN_BUDGET = 12000;

  return (
    <>
      <div className="debug-overlay-bg" onClick={() => setOpen(false)} />
      <aside className="debug-panel">
        <div className="debug-panel-header">
          <span className="debug-panel-title">Debug Panel</span>
          <button className="debug-close-btn" onClick={() => setOpen(false)}>✕</button>
        </div>

        {!lastResult ? (
          <div className="debug-empty">Run an agentic query to see debug info here.</div>
        ) : (
          <div className="debug-body">
            {/* QUERY */}
            <section className="debug-section">
              <div className="debug-section-label">Query</div>
              <div className="debug-query-text">"{lastResult.query}"</div>
            </section>

            {/* STATUS */}
            <section className="debug-section">
              <div className="debug-section-label">Status</div>
              <div className={`debug-status-badge ${lastResult.degraded ? "degraded" : "ok"}`}>
                {lastResult.degraded ? "⚠ Degraded" : "✓ Verified"}
              </div>
              {lastResult.degraded && lastResult.degrade_reason && (
                <div className="debug-degrade-reason">
                  <span className="debug-degrade-code">{lastResult.degrade_reason}</span>
                  <div className="debug-degrade-explain">
                    {DEGRADE_EXPLANATIONS[lastResult.degrade_reason] ?? "Unknown degrade reason."}
                  </div>
                </div>
              )}
            </section>

            {/* LOOP STATS */}
            <section className="debug-section">
              <div className="debug-section-label">Loop Stats</div>
              <div className="debug-stat-grid">
                <div className="debug-stat">
                  <div className="debug-stat-val">{lastResult.iterations_used}</div>
                  <div className="debug-stat-lbl">iterations</div>
                </div>
                <div className="debug-stat">
                  <div className="debug-stat-val">{lastResult.validation_retries}</div>
                  <div className="debug-stat-lbl">verify retries</div>
                </div>
                <div className="debug-stat">
                  <div className="debug-stat-val">{lastResult.sources_count}</div>
                  <div className="debug-stat-lbl">sources used</div>
                </div>
              </div>
              {/* Token budget bar */}
              <div className="debug-budget-row">
                <span className="debug-budget-label">Token budget</span>
                <span className="debug-budget-val">
                  {lastResult.token_budget_used} / {MAX_TOKEN_BUDGET}
                </span>
              </div>
              <div className="debug-budget-bar-bg">
                <div
                  className="debug-budget-bar-fill"
                  style={{
                    width: `${Math.min(100, (lastResult.token_budget_used / MAX_TOKEN_BUDGET) * 100)}%`,
                    background: lastResult.token_budget_used > MAX_TOKEN_BUDGET * 0.8
                      ? "var(--danger)"
                      : "var(--teal)",
                  }}
                />
              </div>
            </section>

            {/* AGENT DECISIONS */}
            <section className="debug-section">
              <div className="debug-section-label">Agent Decisions</div>
              {lastResult.action_steps.length === 0 ? (
                <div className="debug-empty-sub">No steps recorded.</div>
              ) : (
                <div className="debug-steps-list">
                  {lastResult.action_steps.map((step, i) => (
                    <ActionStepRow key={i} step={step} />
                  ))}
                </div>
              )}
            </section>

            {/* CACHE STATUS */}
            {health && (
              <section className="debug-section">
                <div className="debug-section-label">Cache Status</div>
                <div className="debug-cache-grid">
                  <div className="debug-cache-item">
                    <span className="debug-cache-icon" style={{ color: health.query_cache_size > 0 ? "var(--teal)" : "var(--text-muted)" }}>●</span>
                    <span className="debug-cache-label">Query cache</span>
                    <span className="debug-cache-val">{health.query_cache_size} entries</span>
                  </div>
                  <div className="debug-cache-item">
                    <span className="debug-cache-icon" style={{ color: health.retrieval_cache_size > 0 ? "var(--teal)" : "var(--text-muted)" }}>●</span>
                    <span className="debug-cache-label">Retrieval cache</span>
                    <span className="debug-cache-val">{health.retrieval_cache_size} entries</span>
                  </div>
                  <div className="debug-cache-item">
                    <span className="debug-cache-icon" style={{ color: health.sparse_index_built ? "var(--teal)" : "var(--amber)" }}>●</span>
                    <span className="debug-cache-label">BM25 index</span>
                    <span className="debug-cache-val">{health.sparse_corpus_size} docs</span>
                  </div>
                </div>
              </section>
            )}
          </div>
        )}
      </aside>
    </>
  );
}
