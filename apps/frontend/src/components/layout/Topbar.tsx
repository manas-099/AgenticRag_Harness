/**
 * Topbar.tsx — brand mark + live "backend healthy" / "N chunks indexed"
 * status pills, matching the mockup exactly. Polls GET /v1/health every
 * 15s; a failed poll flips the pill to the "warn" (red) dot state.
 */
import { useEffect, useState } from "react";
import { fetchBackendHealth } from "@/api/healthApi";
import type { BackendHealth } from "@/types/domain";

const POLL_INTERVAL_MS = 15_000;

export function Topbar() {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [isHealthy, setIsHealthy] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const result = await fetchBackendHealth();
        if (!cancelled) {
          setHealth(result);
          setIsHealthy(true);
        }
      } catch {
        if (!cancelled) setIsHealthy(false);
      }
    }

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-mark">RH</div>
        <div className="brand-text">
          <span className="name">RAG Harness Studio</span>
          <span className="sub">langgraph · qdrant · fastapi</span>
        </div>
      </div>

      <div className="top-status">
        <span className="status-pill">
          <span className={`status-dot ${isHealthy ? "" : "warn"}`} />
          {isHealthy ? "backend healthy" : "backend unreachable"}
        </span>
        <span className="status-pill hide-mobile">
          <span className="status-dot" />
          {health ? `${health.qdrant_chunk_count} chunks indexed` : "…"}
        </span>
      </div>
    </div>
  );
}
