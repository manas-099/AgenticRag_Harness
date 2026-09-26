/**
 * StepGraph.tsx
 * ---------------------------------------------------------------------------
 * The circle-and-arrow pipeline visualization you asked for specifically:
 *   (circle) --arrow--> (circle) --arrow--> (circle) --arrow--> (circle)
 * Each circle is a pipeline stage (retrieve / evaluate / re-retrieve /
 * synthesize). Circle states:
 *   pending -> gray ring, dim label
 *   active  -> amber ring + a pulsing expanding halo (CSS ::after in
 *              app.css, `.step-node-circle.active::after`) — this is the
 *              "agent is actually doing work right now" effect.
 *   done    -> teal ring, filled dim background
 *   error   -> red ring (used when the run degraded)
 * Arrows between an active/pending boundary get a moving light-sweep
 * animation (`.step-arrow.flowing`) so the whole thing reads as a live
 * pipeline, not a static diagram. Arrows behind a completed stage turn
 * solid teal.
 */
import type { AgentStep } from "@/types/domain";

const STAGE_ORDER: AgentStep["kind"][] = ["retrieve", "evaluate", "reretrieve", "synthesize"];
const STAGE_ICON: Record<AgentStep["kind"], string> = {
  retrieve: "R",
  evaluate: "E",
  reretrieve: "R↻",
  synthesize: "S",
  answer: "A",
};

interface StepGraphProps {
  steps: AgentStep[];
}

/** Reduces the (possibly repeated, e.g. multiple retrieve/evaluate cycles)
 *  step list down to one representative status per pipeline stage, so the
 *  graph always shows exactly 4 nodes regardless of how many search/evaluate
 *  loops the agent actually ran (the full loop detail lives in StepTracker,
 *  which lists every raw step). */
function collapseToStages(steps: AgentStep[]): Record<AgentStep["kind"], AgentStep["status"]> {
  const result: Record<string, AgentStep["status"]> = {
    retrieve: "pending",
    evaluate: "pending",
    reretrieve: "pending",
    synthesize: "pending",
  };
  for (const step of steps) {
    const kind = step.kind === "answer" ? "synthesize" : step.kind;
    // Later occurrences win — so the graph reflects the most recent pass.
    result[kind] = step.status;
  }
  return result as Record<AgentStep["kind"], AgentStep["status"]>;
}

export function StepGraph({ steps }: StepGraphProps) {
  const stageStatus = collapseToStages(steps);
  const activeDetail = steps.find((s) => s.status === "active" || s.status === "error")?.detail;
  const activeLabel = steps.find((s) => s.status === "active" || s.status === "error");

  return (
    <div className="step-graph-wrap">
      <div className="step-graph">
        {STAGE_ORDER.map((kind, i) => {
          const status = stageStatus[kind];
          const isLast = i === STAGE_ORDER.length - 1;
          const nextStatus = !isLast ? stageStatus[STAGE_ORDER[i + 1]] : undefined;

          return (
            <div key={kind} style={{ display: "flex", alignItems: "center", flex: isLast ? "0 0 auto" : 1 }}>
              <div className="step-node">
                <div className={`step-node-circle ${status}`}>{STAGE_ICON[kind]}</div>
                <div className={`step-node-label ${status}`}>{kind === "reretrieve" ? "re-retrieve" : kind}</div>
              </div>
              {!isLast && (
                <div
                  className={`step-arrow ${status === "done" ? "done" : ""} ${
                    status === "active" || nextStatus === "active" ? "flowing" : ""
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>

      {activeLabel && (
        <div className="step-detail">
          <div className="step-detail-title">{activeLabel.label}</div>
          {activeDetail ?? "working…"}
        </div>
      )}
    </div>
  );
}
