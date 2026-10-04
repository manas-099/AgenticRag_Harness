// =============================================================================
// FILE: src/components/chat/ModeToggle.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/chat/ModeToggle.tsx
// NEW FILE
//
// WHAT IT DOES:
//   Two-button toggle: Standard RAG | Agentic RAG
//   Sits at the top of the Chat tab, above the chat window.
//   Standard = single-pass retrieval + generation (fast, no steps shown)
//   Agentic = full ReAct loop (slower, shows step panel)
//
// WHERE IT CONNECTS:
//   store/chatStore.ts (mode, setMode)
// =============================================================================

import { useChatStore } from "@/store/chatStore";

export function ModeToggle() {
  const { mode, setMode } = useChatStore();

  return (
    <div className="mode-toggle">
      <button
        className={`mode-btn ${mode === "standard" ? "active" : ""}`}
        onClick={() => setMode("standard")}
      >
        Standard
      </button>
      <button
        className={`mode-btn ${mode === "agentic" ? "active" : ""}`}
        onClick={() => setMode("agentic")}
      >
        Agentic
      </button>
    </div>
  );
}
