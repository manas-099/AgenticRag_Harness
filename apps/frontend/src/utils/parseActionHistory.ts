/**
 * parseActionHistory.ts
 * ---------------------------------------------------------------------------
 * The backend's agentic response gives us `action_history`, a flat list like
 *   ["search_documents", "search_documents_reformulated", "answer"]
 * This maps that onto the 4-stage pipeline shown in the mockup:
 *   retrieve -> evaluate -> re-retrieve -> synthesize
 *
 * Mapping rules:
 *   - first "search_documents*"                -> retrieve
 *   - any "search_documents_reformulated" after -> evaluate (implicit: the
 *     agent judged the previous result insufficient) followed by re-retrieve
 *   - "answer"                                  -> synthesize
 * This is a heuristic, not something the backend labels explicitly — if you
 * want ground-truth step labels, the cleanest backend addition is to have
 * agent_decide_node include `decision.thought` in the response so this file
 * can show the model's actual reasoning per step instead of an inferred one.
 */
import type { AgentStep } from "@/types/domain";

export function parseActionHistory(actionHistory: string[], degraded: boolean): AgentStep[] {
  const steps: AgentStep[] = [];
  let sawFirstSearch = false;

  actionHistory.forEach((action, i) => {
    const isLast = i === actionHistory.length - 1;

    if (action.startsWith("search_documents")) {
      if (!sawFirstSearch) {
        steps.push({ id: `retrieve-${i}`, kind: "retrieve", label: "retrieve", status: "done" });
        sawFirstSearch = true;
      } else {
        steps.push({
          id: `evaluate-${i}`,
          kind: "evaluate",
          label: "evaluate",
          status: "done",
          detail: "previous chunks insufficient — reformulating query",
        });
        steps.push({ id: `reretrieve-${i}`, kind: "reretrieve", label: "re-retrieve", status: "done" });
      }
    } else if (action === "answer" || isLast) {
      steps.push({
        id: `synthesize-${i}`,
        kind: "synthesize",
        label: "synthesize",
        status: degraded ? "error" : "done",
        detail: degraded ? "answer degraded — some claims dropped" : undefined,
      });
    }
  });

  if (steps.length === 0) {
    steps.push({ id: "retrieve-0", kind: "retrieve", label: "retrieve", status: "pending" });
  }

  return steps;
}

/** Builds a step list for the in-flight (still running) state — every step
 *  up to the current one is "done", the current one is "active", the rest
 *  are "pending". Used while waiting for the API response. */
export function buildInFlightSteps(activeIndex: number): AgentStep[] {
  const kinds: { kind: AgentStep["kind"]; label: string }[] = [
    { kind: "retrieve", label: "retrieve" },
    { kind: "evaluate", label: "evaluate" },
    { kind: "reretrieve", label: "re-retrieve" },
    { kind: "synthesize", label: "synthesize" },
  ];
  return kinds.map((k, i) => ({
    id: `${k.kind}-inflight`,
    kind: k.kind,
    label: k.label,
    status: i < activeIndex ? "done" : i === activeIndex ? "active" : "pending",
  }));
}
