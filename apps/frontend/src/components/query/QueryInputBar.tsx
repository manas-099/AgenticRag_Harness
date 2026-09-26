/**
 * QueryInputBar.tsx — the bottom text input + Run button, matching the
 * mockup's amber "Run" button. Enter submits; Shift+Enter would insert a
 * newline if we ever swap the input for a textarea (kept as a single-line
 * input for now, matching the mockup).
 */
import { useQueryStore } from "@/store/queryStore";

export function QueryInputBar() {
  const question = useQueryStore((s) => s.question);
  const setQuestion = useQueryStore((s) => s.setQuestion);
  const run = useQueryStore((s) => s.run);
  const isRunning = useQueryStore((s) => s.isRunning);

  return (
    <div className="query-input-bar">
      <input
        className="query-input"
        placeholder="Ask a question about your ingested documents…"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !isRunning) void run();
        }}
      />
      <button className="run-btn" onClick={() => void run()} disabled={isRunning || !question.trim()}>
        {isRunning ? "Running…" : "Run"}
      </button>
    </div>
  );
}
