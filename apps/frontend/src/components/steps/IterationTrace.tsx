// =============================================================================
// FILE: src/components/steps/IterationTrace.tsx
// TARGET PATH: apps/frontend/src/components/steps/IterationTrace.tsx
// NEW FILE — replaces StepTracker flat list
//
// WHAT IT DOES:
//   Grouped iteration trace UI showing each ReAct loop iteration as a card:
//   ┌─ Iteration 1 ──────────────┐
//   │ ● retrieve  → 8 chunks     │
//   │ ● evaluate  → searching    │
//   └────────────────────────────┘
//   Color-coded dots:
//     Blue  (#60a5fa) = retrieve / re-retrieve
//     Amber (var(--amber)) = evaluate
//     Teal  (var(--teal)) = synthesize (final)
//     Red   (var(--danger)) = error/degraded
//
//   Header shows the detective cartoon SVG + "Agent Trace" title.
//   The panel has a collapse button (✕ / →) passed from parent.
// =============================================================================

import type { AgentStep } from "@/types/domain";

interface IterationGroup {
  number: number;
  steps: AgentStep[];
}

function groupIntoIterations(steps: AgentStep[]): IterationGroup[] {
  const groups: IterationGroup[] = [];
  let current: IterationGroup | null = null;

  for (const step of steps) {
    if (step.kind === "retrieve" || (step.kind === "reretrieve" && !current)) {
      if (current) groups.push(current);
      current = { number: groups.length + 1, steps: [step] };
    } else if (step.kind === "reretrieve") {
      if (current) groups.push(current);
      current = { number: groups.length + 1, steps: [step] };
    } else {
      if (!current) current = { number: 1, steps: [] };
      current.steps.push(step);
    }
  }
  if (current && current.steps.length > 0) groups.push(current);
  return groups;
}

const STEP_COLOR: Record<string, string> = {
  retrieve: "#60a5fa",
  reretrieve: "#60a5fa",
  evaluate: "var(--amber)",
  synthesize: "var(--teal)",
  answer: "var(--teal)",
};

const STEP_LABEL: Record<string, string> = {
  retrieve: "retrieve",
  reretrieve: "re-retrieve",
  evaluate: "evaluate",
  synthesize: "synthesize",
  answer: "answer",
};

function StepDot({ kind, status }: { kind: string; status: AgentStep["status"] }) {
  const color = status === "error" ? "var(--danger)" : STEP_COLOR[kind] ?? "var(--text-muted)";
  const isActive = status === "active";
  return (
    <span
      className={`iter-dot ${isActive ? "active" : ""}`}
      style={{ background: color, boxShadow: isActive ? `0 0 0 4px ${color}33` : "none" }}
    />
  );
}

// Inline SVG detective cartoon (simplified, no external image needed)
function DetectiveIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Hat brim */}
      <ellipse cx="18" cy="11" rx="11" ry="3" fill="#c97a1f" opacity="0.9"/>
      {/* Hat top */}
      <rect x="11" y="4" width="14" height="8" rx="3" fill="#c97a1f"/>
      {/* Hat band */}
      <rect x="10" y="10" width="16" height="2" fill="#e8a33d"/>
      {/* Face */}
      <circle cx="18" cy="20" r="7" fill="#fde68a"/>
      {/* Eyes */}
      <circle cx="15.5" cy="19" r="1.2" fill="#1e293b"/>
      <circle cx="20.5" cy="19" r="1.2" fill="#1e293b"/>
      {/* Eye shine */}
      <circle cx="16" cy="18.5" r="0.4" fill="white"/>
      <circle cx="21" cy="18.5" r="0.4" fill="white"/>
      {/* Magnifying glass */}
      <circle cx="26" cy="26" r="4" stroke="#e8a33d" strokeWidth="1.5" fill="none"/>
      <circle cx="26" cy="26" r="2" fill="#bfdbfe" opacity="0.5"/>
      <line x1="23" y1="29" x2="21" y2="31" stroke="#e8a33d" strokeWidth="1.5" strokeLinecap="round"/>
      {/* Pipe */}
      <path d="M14 23 Q12 24 11 23 Q10 22 11 21" stroke="#92400e" strokeWidth="1.2" fill="none" strokeLinecap="round"/>
      {/* Smoke */}
      <path d="M10 20 Q9 18 10 17" stroke="#94a3b8" strokeWidth="0.8" fill="none" opacity="0.6"/>
    </svg>
  );
}

interface IterationTraceProps {
  steps: AgentStep[];
  onClose?: () => void;
}

export function IterationTrace({ steps, onClose }: IterationTraceProps) {
  const groups = groupIntoIterations(steps);
  const isRunning = steps.some((s) => s.status === "active");
  const isDone = !isRunning && steps.length > 0;

  return (
    <div className="iter-trace">
      {/* Header */}
      <div className="iter-trace-header">
        <div className="iter-trace-title-row">
          <DetectiveIcon />
          <div>
            <div className="iter-trace-title">Agent Trace</div>
            <div className="iter-trace-subtitle">
              {isRunning ? "investigating…" : isDone ? `${groups.length} iteration${groups.length !== 1 ? "s" : ""}` : "waiting"}
            </div>
          </div>
        </div>
        {onClose && (
          <button className="iter-close-btn" onClick={onClose} title="Close trace">✕</button>
        )}
      </div>

      {/* Color legend */}
      <div className="iter-legend">
        <span className="iter-legend-item"><span className="iter-dot-sm" style={{ background: "#60a5fa" }} />retrieve</span>
        <span className="iter-legend-item"><span className="iter-dot-sm" style={{ background: "var(--amber)" }} />evaluate</span>
        <span className="iter-legend-item"><span className="iter-dot-sm" style={{ background: "var(--teal)" }} />synthesize</span>
      </div>

      {/* Iteration groups */}
      <div className="iter-groups">
        {groups.length === 0 && (
          <div className="iter-empty">Run an agentic query to see the trace.</div>
        )}
        {groups.map((group) => (
          <div className="iter-group" key={group.number}>
            <div className="iter-group-label">Iteration {group.number}</div>
            <div className="iter-group-steps">
              {group.steps.map((step) => (
                <div className="iter-step-row" key={step.id}>
                  <StepDot kind={step.kind} status={step.status} />
                  <span className="iter-step-name" style={{ color: step.status === "error" ? "var(--danger)" : STEP_COLOR[step.kind] }}>
                    {STEP_LABEL[step.kind] ?? step.label}
                  </span>
                  {step.detail && (
                    <span className="iter-step-detail">{step.detail}</span>
                  )}
                  {step.status === "active" && (
                    <span className="iter-step-detail" style={{ color: "var(--amber)" }}>searching…</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
