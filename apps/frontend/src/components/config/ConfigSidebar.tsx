// ConfigSidebar.tsx
// Stripped to LLM config only. Resizable via drag handle on right edge.

import { useState, useRef, useCallback } from "react";
import { useLLMConfigStore } from "@/store/llmConfigStore";
import { LLM_PROVIDERS, type LLMProvider } from "@/types/llmConfig";
import { useEffect } from "react";
import { fetchLLMHealth } from "@/api/healthApi";
import type { LLMHealth } from "@/types/domain";

interface ConfigSidebarProps {
  open: boolean;
  onToggle: () => void;
}

export function ConfigSidebar({ open, onToggle }: ConfigSidebarProps) {
  const { mode, custom, setMode, updateCustom } = useLLMConfigStore();
  const [llmHealth, setLlmHealth] = useState<LLMHealth | null>(null);
  const [width, setWidth] = useState(220);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startW = useRef(0);

  useEffect(() => {
    fetchLLMHealth().then(setLlmHealth).catch(() => {});
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    startX.current = e.clientX;
    startW.current = width;
    document.body.style.userSelect = "none";

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const delta = ev.clientX - startX.current;
      setWidth(Math.max(160, Math.min(400, startW.current + delta)));
    };
    const onUp = () => {
      dragging.current = false;
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [width]);

  if (!open) {
    return (
      <aside className="csb-rail">
        <button className="csb-rail-btn" onClick={onToggle} title="Expand config">
          <span>⚙</span>
        </button>
        <div className="csb-rail-divider" />
        <div className="csb-rail-icon" title="Model">💾</div>
      </aside>
    );
  }

  return (
    <aside className="csb-panel" style={{ width }}>
      {/* Header */}
      <div className="csb-header">
        <div className="csb-header-title">
          <span className="csb-header-icon">⚙</span>
          <span>Config</span>
        </div>
        <button className="csb-collapse-btn" onClick={onToggle} title="Collapse">⟨</button>
      </div>

      <div className="csb-body">
        {/* Model / LLM */}
        <div className="csb-section-head">
          <span className="csb-section-icon">💾</span>
          <span className="csb-section-label">Model</span>
        </div>

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
            <div className="csb-row">
              <span className="csb-row-label">Model</span>
              <input
                className="csb-row-input"
                value={custom.model}
                onChange={(e) => updateCustom({ model: e.target.value })}
                placeholder="model name"
              />
            </div>
            <div className="csb-row">
              <span className="csb-row-label">API Key</span>
              <input
                className="csb-row-input"
                type="password"
                value={custom.apiKey}
                onChange={(e) => updateCustom({ apiKey: e.target.value })}
                placeholder="sk-…"
              />
            </div>
          </>
        )}

        {/* Chain Status */}
        {llmHealth && (
          <>
            <div className="csb-section-head" style={{ marginTop: 12 }}>
              <span className="csb-section-icon">📡</span>
              <span className="csb-section-label">Chain Status</span>
            </div>
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

      {/* Drag handle */}
      <div className="csb-drag-handle" onMouseDown={onMouseDown} />
    </aside>
  );
}
