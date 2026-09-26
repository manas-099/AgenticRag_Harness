/**
 * ModeTabs.tsx — Standard / Agentic / Compare switch, matching the mockup.
 */
import { useQueryStore, type QueryMode } from "@/store/queryStore";

const MODES: { id: QueryMode; label: string }[] = [
  { id: "standard", label: "Standard" },
  { id: "agentic", label: "Agentic" },
  { id: "compare", label: "Compare" },
];

export function ModeTabs() {
  const mode = useQueryStore((s) => s.mode);
  const setMode = useQueryStore((s) => s.setMode);

  return (
    <div className="mode-tabs">
      {MODES.map((m) => (
        <button
          key={m.id}
          className={`mode-tab ${mode === m.id ? "active" : ""}`}
          onClick={() => setMode(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
