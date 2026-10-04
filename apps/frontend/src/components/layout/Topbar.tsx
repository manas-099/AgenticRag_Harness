// =============================================================================
// FILE: src/components/layout/Topbar.tsx
// TARGET PATH: apps/frontend/src/components/layout/Topbar.tsx
// REPLACES: previous Topbar.tsx
//
// WHAT CHANGED:
//   - Tab nav now has only 2 tabs: Ingest | Chat (Config moved to left sidebar)
//   - Book+animals SVG illustration added next to brand name in header
//   - Debug toggle button kept
// =============================================================================

import { useEffect, useState } from "react";
import type { AppTab } from "@/App";
import { useDebugStore } from "@/store/debugStore";
import { fetchBackendHealth } from "@/api/healthApi";

interface TopbarProps {
  activeTab: AppTab;
  onTabChange: (tab: AppTab) => void;
}

// Inline book+animals SVG (simplified watercolor-style)
function BookCharacter() {
  return (
    <svg width="38" height="30" viewBox="0 0 38 30" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Book body */}
      <path d="M4 6 Q4 4 6 4 L18 5 L30 4 Q32 4 32 6 L32 26 Q32 28 30 28 L6 28 Q4 28 4 26 Z" fill="#1e3a5f" opacity="0.9"/>
      {/* Book spine */}
      <rect x="17" y="4" width="2" height="24" fill="#0f2744" opacity="0.7"/>
      {/* Left page */}
      <rect x="5" y="6" width="12" height="20" rx="1" fill="#fef9e7" opacity="0.95"/>
      {/* Right page */}
      <rect x="19" y="6" width="12" height="20" rx="1" fill="#fefce8" opacity="0.95"/>
      {/* Small fox on left page */}
      <ellipse cx="10" cy="20" rx="3" ry="2" fill="#f97316" opacity="0.8"/>
      <path d="M8 18 L7 16 L9 17 Z" fill="#f97316" opacity="0.8"/>
      <path d="M12 18 L13 16 L11 17 Z" fill="#f97316" opacity="0.8"/>
      <circle cx="10" cy="19" r="1.2" fill="#fed7aa"/>
      {/* Small cat on right page */}
      <ellipse cx="25" cy="20" rx="2.5" ry="2" fill="#94a3b8" opacity="0.8"/>
      <path d="M23 18.5 L22.5 17 L24 18 Z" fill="#94a3b8" opacity="0.8"/>
      <path d="M27 18.5 L27.5 17 L26 18 Z" fill="#94a3b8" opacity="0.8"/>
      <circle cx="25" cy="19.5" r="1" fill="#e2e8f0"/>
      {/* Small deer on right page */}
      <ellipse cx="28" cy="16" rx="1.5" ry="1.2" fill="#d97706" opacity="0.7"/>
      {/* Antlers */}
      <path d="M27.5 15 L27 13 M28.5 15 L29 13" stroke="#92400e" strokeWidth="0.6"/>
    </svg>
  );
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
    { id: "ingest", label: "Ingest" },
    { id: "chat", label: "Chat" },
  ];

  return (
    <header className="topbar">
      <div className="brand">
        <BookCharacter />
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
        <button className="debug-toggle-btn" onClick={toggleOpen} title="Open Debug Panel">
          <span className="debug-icon">⬡</span>
          {hasWarn && <span className="debug-dot warn" />}
        </button>
        <div className="status-pill">
          <span className={`status-dot ${healthy === false ? "warn" : ""}`} />
          <span className="hide-mobile">
            {healthy === null ? "connecting…" : healthy ? "backend ok" : "offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
