/**
 * QueryWorkspace.tsx — Zone B (center column). Composes:
 *   ModeTabs -> HistoryStrip -> [StepGraph + StepTracker while
 *   agentic] -> ResultCard -> QueryInputBar
 * StepGraph/StepTracker only render meaningfully in "agentic" mode — a
 * standard-mode run has no multi-step trace, so they're hidden then.
 */
import { useQueryStore } from "@/store/queryStore";
import { ModeTabs } from "@/components/query/ModeTabs";
import { QueryInputBar } from "@/components/query/QueryInputBar";
import { ResultCard } from "@/components/query/ResultCard";
import { HistoryStrip } from "@/components/history/HistoryStrip";
import { StepGraph } from "@/components/steps/StepGraph";
import { StepTracker } from "@/components/steps/StepTracker";

export function QueryWorkspace() {
  const mode = useQueryStore((s) => s.mode);
  const steps = useQueryStore((s) => s.steps);
  const showSteps = mode !== "standard" && steps.length > 0;

  return (
    <div className="zone zone-b">
      <ModeTabs />
      <HistoryStrip />
      {showSteps && (
        <>
          <StepGraph steps={steps} />
          <StepTracker steps={steps} />
        </>
      )}
      <div className="query-body">
        <ResultCard />
      </div>
      <QueryInputBar />
    </div>
  );
}
