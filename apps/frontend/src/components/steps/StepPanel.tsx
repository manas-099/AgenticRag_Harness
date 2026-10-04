// =============================================================================
// FILE: src/components/steps/StepPanel.tsx
// TARGET PATH: apps/frontend/src/components/steps/StepPanel.tsx
// REPLACES: previous StepPanel.tsx
//
// WHAT CHANGED:
//   - Uses new IterationTrace instead of StepGraph + StepTracker
//   - Panel has open/close toggle (→ button when closed, ✕ inside IterationTrace)
//   - When closed: shows a thin 40px rail with a ⬡ icon — click to reopen
//   - stepPanelOpen state lives in chatStore
// =============================================================================

import { useChatStore } from "@/store/chatStore";
import { IterationTrace } from "@/components/steps/IterationTrace";
import { StepGraph } from "@/components/steps/StepGraph";

export function StepPanel() {
  const { steps, isRunning, stepPanelOpen, setStepPanelOpen } = useChatStore((s) => ({
    steps: s.steps,
    isRunning: s.isRunning,
    stepPanelOpen: s.stepPanelOpen,
    setStepPanelOpen: s.setStepPanelOpen,
  }));

  // Collapsed rail
  if (!stepPanelOpen) {
    return (
      <div className="step-panel-rail" onClick={() => setStepPanelOpen(true)} title="Open agent trace">
        <span className="step-rail-icon">⬡</span>
        {isRunning && <span className="step-rail-pulse" />}
      </div>
    );
  }

  return (
    <div className="step-panel">
      {/* Circle graph stays at top */}
      {steps.length > 0 && (
        <div className="step-graph-section">
          <StepGraph steps={steps} />
        </div>
      )}
      {/* New grouped iteration trace */}
      <IterationTrace steps={steps} onClose={() => setStepPanelOpen(false)} />
    </div>
  );
}
