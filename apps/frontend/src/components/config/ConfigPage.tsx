// =============================================================================
// FILE: src/components/config/ConfigPage.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/config/ConfigPage.tsx
// NEW FILE (replaces ConfigPanel.tsx in old zone-C)
//
// WHAT IT DOES:
//   Full-page config tab. Two sections:
//
//   1. LLM CHAIN CONFIG — same Default/Custom toggle from old ConfigPanel.
//      Default: backend's .env chain (MyLLM → OpenRouter → Groq)
//      Custom: user picks provider + model + API key (BYOK)
//      Notes say "(1) one config tab where LLM and their API URL will be set"
//
//   2. LLM CHAIN STATUS — live from GET /v1/health/llm, shows:
//      Agent chain: [myllm] → [groq], last used: groq
//      Generate chain: [myllm] → [openrouter] → [groq], last used: myllm
//      Each backend pill is green if reachable, red if last_backend_used was
//      a fallback.
//      Notes say "(2)(d) tab - that only do injecting ones that already in backend"
//
// WHY:
//   Keeps all LLM config in one dedicated tab. LLM health gives the user
//   visibility into which backend is actually answering their queries.
//
// WHERE IT CONNECTS:
//   store/llmConfigStore.ts (mode, custom, setMode, updateCustom — unchanged)
//   api/healthApi.ts (fetchLLMHealth)
//   types/llmConfig.ts (LLM_PROVIDERS — unchanged)
// =============================================================================

import { useEffect, useState } from "react";
import { useLLMConfigStore } from "@/store/llmConfigStore";
import { LLM_PROVIDERS, type LLMProvider } from "@/types/llmConfig";
import { fetchLLMHealth } from "@/api/healthApi";
import type { LLMHealth } from "@/types/domain";

export function ConfigPage() {
  const { mode, custom, setMode, updateCustom } = useLLMConfigStore();
  const [justSaved, setJustSaved] = useState(false);
  const [llmHealth, setLlmHealth] = useState<LLMHealth | null>(null);

  useEffect(() => {
    fetchLLMHealth().then(setLlmHealth).catch(() => {});
    const t = setInterval(() => fetchLLMHealth().then(setLlmHealth).catch(() => {}), 30_000);
    return () => clearInterval(t);
  }, []);

  function handleProviderChange(provider: LLMProvider) {
    const preset = LLM_PROVIDERS.find((p) => p.id === provider);
    updateCustom({ provider, model: preset?.defaultModel ?? "" });
  }

  function handleSave() {
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 2000);
  }

  return (
    <div className="config-page">
      <div className="config-page-inner">

        {/* LLM Config */}
        <section className="config-section">
          <div className="config-section-label">LLM Configuration</div>

          <div className="config-toggle-row">
            <button
              className={`config-toggle-option ${mode === "default" ? "active" : ""}`}
              onClick={() => setMode("default")}
            >
              Default (backend .env)
            </button>
            <button
              className={`config-toggle-option ${mode === "custom" ? "active" : ""}`}
              onClick={() => setMode("custom")}
            >
              Custom (BYOK)
            </button>
          </div>

          {mode === "default" ? (
            <p className="config-hint">
              Using the backend's configured chain: MyLLM → OpenRouter → Groq (from .env).
              No credentials leave this browser.
            </p>
          ) : (
            <>
              <p className="config-hint">
                Override the backend chain for every request. Your key is stored in this
                browser only and sent as request headers.
              </p>
              <div className="config-field">
                <label htmlFor="cfg-provider">Provider</label>
                <select
                  id="cfg-provider"
                  value={custom.provider}
                  onChange={(e) => handleProviderChange(e.target.value as LLMProvider)}
                >
                  {LLM_PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>{p.label}</option>
                  ))}
                </select>
              </div>
              <div className="config-field">
                <label htmlFor="cfg-model">Model</label>
                <input
                  id="cfg-model"
                  value={custom.model}
                  onChange={(e) => updateCustom({ model: e.target.value })}
                  placeholder="e.g. gpt-4o-mini"
                />
              </div>
              {custom.provider === "custom" && (
                <div className="config-field">
                  <label htmlFor="cfg-base-url">Base URL</label>
                  <input
                    id="cfg-base-url"
                    value={custom.baseUrl ?? ""}
                    onChange={(e) => updateCustom({ baseUrl: e.target.value })}
                    placeholder="https://your-endpoint/v1/chat/completions"
                  />
                </div>
              )}
              <div className="config-field">
                <label htmlFor="cfg-key">API Key</label>
                <input
                  id="cfg-key"
                  type="password"
                  value={custom.apiKey}
                  onChange={(e) => updateCustom({ apiKey: e.target.value })}
                  placeholder="sk-…"
                />
              </div>
              <button className="config-save-btn" onClick={handleSave}>Save</button>
              {justSaved && <div className="config-status">✓ saved to browser storage</div>}
            </>
          )}
        </section>

        {/* LLM Chain Status */}
        {llmHealth && (
          <section className="config-section">
            <div className="config-section-label">Live Chain Status</div>
            {(["agent", "generate"] as const).map((role) => {
              const chain = llmHealth[role];
              return (
                <div className="chain-status-card" key={role}>
                  <div className="chain-role">{role === "agent" ? "Agent (decide)" : "Generate (answer)"}</div>
                  <div className="chain-order">
                    {chain.order.map((backend, i) => {
                      const isLast = chain.last_backend_used === backend;
                      return (
                        <span key={backend} className="chain-backend-row">
                          <span className={`chain-backend-pill ${isLast ? "active" : ""}`}>
                            {backend}
                          </span>
                          {i < chain.order.length - 1 && <span className="chain-arrow">→</span>}
                        </span>
                      );
                    })}
                  </div>
                  {chain.last_backend_used && (
                    <div className="chain-last-used">Last used: <strong>{chain.last_backend_used}</strong></div>
                  )}
                  {chain.myllm_reachable === false && (
                    <div className="chain-warn">MyLLM unreachable — falling back to cloud</div>
                  )}
                </div>
              );
            })}
          </section>
        )}
      </div>
    </div>
  );
}
