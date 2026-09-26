/**
 * StepTracker.tsx — the linear "retrieve / evaluate / re-retrieve /
 * synthesize" list from the mockup's left rail of the Agentic RAG card.
 * Shows every raw step (including repeated retrieve/evaluate loops), unlike
 * StepGraph which collapses to 4 stage nodes. The two together = the full
 * picture: StepGraph for the "shape" of the run, StepTracker for the
 * detailed timeline.
 */
import type { AgentStep } from "@/types/domain";

interface StepTrackerProps {
  steps: AgentStep[];
}

export function StepTracker({ steps }: StepTrackerProps) {
  return (
    <div className="trace">
      <div className="trace-label">Agent trace</div>
      {steps.map((step) => (
        <div className={`trace-step ${step.status === "active" ? "live" : step.status}`} key={step.id}>
          <span className="trace-dot" />
          {step.label}
        </div>
      ))}
    </div>
  );
}
