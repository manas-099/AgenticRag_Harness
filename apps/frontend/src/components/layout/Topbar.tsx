// =============================================================================
// FILE: src/components/layout/Topbar.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/layout/Topbar.tsx
// REPLACES: existing Topbar.tsx (full replacement)
//
// WHAT CHANGED:
//   - Tab navigation added: Chat | Ingest | Config pills in the center
//   - Debug Panel toggle button added on the right (bug icon, shows dot
//     indicator when last run was degraded)
//   - Health status dot kept from original
//
// WHY:
//   The new tab-based layout needs global tab switching from the topbar.
//   Debug Panel needs a persistent toggle that doesn't live inside any tab.
//
// WHERE IT CONNECTS:
//   App.tsx passes activeTab + onTabChange
//   store/debugStore.ts provides toggleOpen() + lastResult.degraded for the dot
//   api/healthApi.ts polled every 30s for the status dot
// =============================================================================

import { useEffect, useState } from "react";
import type { AppTab } from "@/App";
import { useDebugStore } from "@/store/debugStore";
import { fetchBackendHealth } from "@/api/healthApi";

interface TopbarProps {
  activeTab: AppTab;
  onTabChange: (tab: AppTab) => void;
}

export function Topbar({ activeTab, onTabChange }: TopbarProps) {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const { toggleOpen, lastResult } = useDebugStore();
  const hasWarn = lastResult?.degraded ?? false;

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        await fetchBackendHealth();
        if (!cancelled) setHealthy(true);
      } catch {
        if (!cancelled) setHealthy(false);
      }
    }
    check();
    const t = setInterval(check, 30_000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  const TABS: { id: AppTab; label: string }[] = [
    { id: "chat", label: "Chat" },
    { id: "ingest", label: "Ingest" },
    { id: "config", label: "Config" },
  ];

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">H</div>
        <div className="brand-text">
          <span className="name">RAG Harness</span>
          <span className="sub">agentic · local</span>
        </div>
      </div>

      <nav className="topbar-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`topbar-tab ${activeTab === t.id ? "active" : ""}`}
            onClick={() => onTabChange(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="top-status">
        {/* Debug panel toggle */}
        <button className="debug-toggle-btn" onClick={toggleOpen} title="Open Debug Panel">
          <span className="debug-icon">⬡</span>
          {hasWarn && <span className="debug-dot warn" />}
        </button>

        {/* Backend health dot */}
        <div className="status-pill">
          <span className={`status-dot ${healthy === false ? "warn" : ""}`} />
          <span className="hide-mobile">
            {healthy === null ? "connecting…" : healthy ? "backend ok" : "backend offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
