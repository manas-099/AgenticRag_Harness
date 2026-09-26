/**
 * ConfigPanel.tsx — Zone C: the "which model answers my questions" toggle
 * you asked for.
 *   Default -> nothing sent; backend uses its own .env-configured chain
 *              (MyLLM -> OpenRouter -> Groq, see backend PATCH_NOTES.md).
 *   Custom  -> user picks a provider + model + API key here; every request
 *              attaches them as X-LLM-* headers (see api/client.ts). Needs
 *              a small backend addition to actually be honored server-side
 *              — see README "Wiring up bring-your-own-key on the backend".
 */
import { useState } from "react";
import { useLLMConfigStore } from "@/store/llmConfigStore";
import { LLM_PROVIDERS, type LLMProvider } from "@/types/llmConfig";

export function ConfigPanel() {
  const { mode, custom, setMode, updateCustom } = useLLMConfigStore();
  const [justSaved, setJustSaved] = useState(false);

  function handleProviderChange(provider: LLMProvider) {
    const preset = LLM_PROVIDERS.find((p) => p.id === provider);
    updateCustom({ provider, model: preset?.defaultModel ?? "" });
  }

  function handleSave() {
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 2000);
  }

  return (
    <div className="config-body">
      <div className="config-toggle-row">
        <button
          className={`config-toggle-option ${mode === "default" ? "active" : ""}`}
          onClick={() => setMode("default")}
        >
          Default
        </button>
        <button
          className={`config-toggle-option ${mode === "custom" ? "active" : ""}`}
          onClick={() => setMode("custom")}
        >
          Custom
        </button>
      </div>

      {mode === "default" ? (
        <p className="config-hint">
          Using the backend's own configured chain (MyLLM → OpenRouter → Groq, as set in its <code>.env</code>).
          No credentials leave this browser.
        </p>
      ) : (
        <>
          <p className="config-hint">
            Bring your own provider + model + API key. Stored only in this browser's local storage, sent to your
            backend as request headers on every query.
          </p>

          <div className="config-field">
            <label htmlFor="cfg-provider">Provider</label>
            <select
              id="cfg-provider"
              value={custom.provider}
              onChange={(e) => handleProviderChange(e.target.value as LLMProvider)}
            >
              {LLM_PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
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
            <label htmlFor="cfg-key">API key</label>
            <input
              id="cfg-key"
              type="password"
              value={custom.apiKey}
              onChange={(e) => updateCustom({ apiKey: e.target.value })}
              placeholder="sk-…"
            />
          </div>

          <button className="config-save-btn" onClick={handleSave}>
            Save
          </button>
          {justSaved && <div className="config-status">✓ saved locally</div>}
        </>
      )}
    </div>
  );
}
