/**
 * HistoryStrip.tsx — horizontal scrollable pills of recent runs, matching
 * the mockup's "now · rollback steps for prod" / "2m ago · ..." tabs.
 */
import { useQueryStore } from "@/store/queryStore";

function timeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 10) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ago`;
}

export function HistoryStrip() {
  const history = useQueryStore((s) => s.history);
  const setQuestion = useQueryStore((s) => s.setQuestion);

  if (history.length === 0) return null;

  return (
    <div className="history-strip">
      {history.map((entry, i) => (
        <button
          key={entry.id}
          className={`history-pill ${i === 0 ? "now" : ""}`}
          onClick={() => setQuestion(entry.question)}
          title={entry.question}
        >
          <span className="ago">{timeAgo(entry.timestamp)}</span>
          {entry.question.length > 34 ? `${entry.question.slice(0, 34)}…` : entry.question}
        </button>
      ))}
    </div>
  );
}
