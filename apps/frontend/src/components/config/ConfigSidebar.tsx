// =============================================================================
// FILE: src/components/config/ConfigSidebar.tsx
// TARGET PATH: apps/frontend/src/components/config/ConfigSidebar.tsx
// NEW FILE
//
// WHAT IT DOES:
//   Left sidebar with collapsible config panel.
//
//   OPEN STATE (240px):
//   ┌──────────────────────┐
//   │ ⚙ Config      [⟨]   │
//   │ ── Retrieval ──────  │
//   │  Top-K      [5]      │
//   │  Threshold  [0.7]    │
//   │ ── Agent ──────────  │
//   │  Max Iter   [3]      │
//   │  Rescore    [0.8]    │
//   │ ── Model ──────────  │
//   │  LLM        [default]│
//   └──────────────────────┘
//
//   CLOSED STATE (44px icon rail):
//   ┌────┐
//   │ ⚙  │  ← click to expand
//   │    │
//   │ 🎯 │
//   │ 🤖 │
//   │ 💾 │
//   └────┘
//   Hovering icons shows tooltip labels.
//
// WHY:
//   Notes and images show left sidebar layout with collapsible config panel.
//   Matches the ASCII wireframe from the user's notes exactly.
//
// WHERE IT CONNECTS:
//   App.tsx (open, onToggle props)
//   store/llmConfigStore.ts (LLM mode/custom config)
// =============================================================================

import { useState } from "react";
import { useLLMConfigStore } from "@/store/llmConfigStore";
import { LLM_PROVIDERS, type LLMProvider } from "@/types/llmConfig";
import { useEffect } from "react";
import { fetchLLMHealth } from "@/api/healthApi";
import type { LLMHealth } from "@/types/domain";

interface ConfigSidebarProps {
  open: boolean;
  onToggle: () => void;
}

// Section header inside sidebar
function SectionHead({ label, icon }: { label: string; icon: string }) {
  return (
    <div className="csb-section-head">
      <span className="csb-section-icon">{icon}</span>
      <span className="csb-section-label">{label}</span>
    </div>
  );
}

// Editable row: label + inline value input
function ConfigRow({ label, value, onChange, type = "text" }: {
  label: string; value: string; onChange: (v: string) => void; type?: string;
}) {
  return (
    <div className="csb-row">
      <span className="csb-row-label">{label}</span>
      <input
        className="csb-row-input"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

export function ConfigSidebar({ open, onToggle }: ConfigSidebarProps) {
  const { mode, custom, setMode, updateCustom } = useLLMConfigStore();
  const [llmHealth, setLlmHealth] = useState<LLMHealth | null>(null);

  // Local retrieval/agent settings (display only — backend uses .env)
  const [topK, setTopK] = useState("5");
  const [threshold, setThreshold] = useState("0.7");
  const [maxIter, setMaxIter] = useState("3");
  const [rescore, setRescore] = useState("0.8");

  useEffect(() => {
    fetchLLMHealth().then(setLlmHealth).catch(() => {});
  }, []);

  // Collapsed icon rail
  if (!open) {
    return (
      <aside className="csb-rail">
        <button className="csb-rail-btn" onClick={onToggle} title="Expand config">
          <span>⚙</span>
        </button>
        <div className="csb-rail-divider" />
        <div className="csb-rail-icon" title="Retrieval">🎯</div>
        <div className="csb-rail-icon" title="Agent">🤖</div>
        <div className="csb-rail-icon" title="Model">💾</div>
      </aside>
    );
  }

  return (
    <aside className="csb-panel">
      {/* Header */}
      <div className="csb-header">
        <div className="csb-header-title">
          <span className="csb-header-icon">⚙</span>
          <span>Config</span>
        </div>
        <button className="csb-collapse-btn" onClick={onToggle} title="Collapse">⟨</button>
      </div>

      <div className="csb-body">
        {/* Retrieval */}
        <SectionHead label="Retrieval" icon="🎯" />
        <ConfigRow label="Top-K" value={topK} onChange={setTopK} />
        <ConfigRow label="Threshold" value={threshold} onChange={setThreshold} />

        {/* Agent */}
        <SectionHead label="Agent" icon="🤖" />
        <ConfigRow label="Max Iter" value={maxIter} onChange={setMaxIter} />
        <ConfigRow label="Rescore" value={rescore} onChange={setRescore} />

        {/* Model */}
        <SectionHead label="Model" icon="💾" />
        <div className="csb-toggle-row">
          <button
            className={`csb-toggle-btn ${mode === "default" ? "active" : ""}`}
            onClick={() => setMode("default")}
          >Default</button>
          <button
            className={`csb-toggle-btn ${mode === "custom" ? "active" : ""}`}
            onClick={() => setMode("custom")}
          >Custom</button>
        </div>

        {mode === "default" ? (
          <div className="csb-hint">Using backend .env chain</div>
        ) : (
          <>
            <div className="csb-row">
              <span className="csb-row-label">Provider</span>
              <select
                className="csb-row-input"
                value={custom.provider}
                onChange={(e) => {
                  const p = e.target.value as LLMProvider;
                  const preset = LLM_PROVIDERS.find((x) => x.id === p);
                  updateCustom({ provider: p, model: preset?.defaultModel ?? "" });
                }}
              >
                {LLM_PROVIDERS.map((p) => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>
            </div>
            <ConfigRow label="Model" value={custom.model} onChange={(v) => updateCustom({ model: v })} />
            <ConfigRow label="API Key" value={custom.apiKey} onChange={(v) => updateCustom({ apiKey: v })} type="password" />
          </>
        )}

        {/* LLM chain status */}
        {llmHealth && (
          <>
            <SectionHead label="Chain Status" icon="📡" />
            {(["agent", "generate"] as const).map((role) => {
              const chain = llmHealth[role];
              return (
                <div className="csb-chain-row" key={role}>
                  <span className="csb-chain-role">{role}</span>
                  <div className="csb-chain-backends">
                    {chain.order.map((b) => (
                      <span
                        key={b}
                        className={`csb-backend-pill ${chain.last_backend_used === b ? "active" : ""}`}
                      >{b}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </>
        )}
      </div>
    </aside>
  );
}
