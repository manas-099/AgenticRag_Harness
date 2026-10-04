// =============================================================================
// FILE: src/components/steps/StepPanel.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/steps/StepPanel.tsx
// NEW FILE
//
// WHAT IT DOES:
//   The right sidebar panel shown during/after an agentic run.
//   Contains two sections stacked vertically:
//     1. StepGraph (circle → arrow → circle pipeline diagram, existing component)
//     2. StepTracker (linear list of every raw step, existing component)
//
//   Panel header shows: "Agent Steps" title + current status badge
//   (running / done / degraded).
//
// WHY:
//   Notes sketch shows "steps" box on the right side of the chat with
//   "Ingest → Retrieval" flow. This is that panel.
//
// WHERE IT CONNECTS:
//   store/chatStore.ts (steps, isRunning)
//   components/steps/StepGraph.tsx (unchanged, reused)
//   components/steps/StepTracker.tsx (unchanged, reused)
// =============================================================================

import { useChatStore } from "@/store/chatStore";
import { StepGraph } from "@/components/steps/StepGraph";
import { StepTracker } from "@/components/steps/StepTracker";

export function StepPanel() {
  const { steps, isRunning } = useChatStore((s) => ({ steps: s.steps, isRunning: s.isRunning }));

  const status = isRunning ? "running" : steps.some((s) => s.status === "error") ? "degraded" : "done";

  return (
    <div className="step-panel">
      <div className="step-panel-header">
        <span className="step-panel-title">Agent Steps</span>
        <span className={`step-panel-status ${status}`}>
          {status === "running" ? "⟳ running" : status === "degraded" ? "⚠ degraded" : "✓ done"}
        </span>
      </div>
      <div className="step-panel-body">
        <StepGraph steps={steps} />
        <StepTracker steps={steps} />
      </div>
    </div>
  );
}
