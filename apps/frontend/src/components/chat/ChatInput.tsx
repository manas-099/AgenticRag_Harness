// =============================================================================
// FILE: src/components/chat/ChatInput.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/chat/ChatInput.tsx
// NEW FILE
//
// WHAT IT DOES:
//   Bottom input bar for the chat. Textarea grows with content (up to 5 rows).
//   Enter sends (Shift+Enter for newline). Send button disabled while isRunning.
//   Clear history button (trash icon) at the right of the bar.
//
// WHERE IT CONNECTS:
//   store/chatStore.ts (send, isRunning, clearHistory)
// =============================================================================

import { useState, useRef, KeyboardEvent } from "react";
import { useChatStore } from "@/store/chatStore";

export function ChatInput() {
  const [text, setText] = useState("");
  const { send, isRunning, clearHistory } = useChatStore();
  const ref = useRef<HTMLTextAreaElement>(null);

  function handleSend() {
    const q = text.trim();
    if (!q || isRunning) return;
    send(q);
    setText("");
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-input-bar">
      <textarea
        ref={ref}
        className="chat-textarea"
        rows={1}
        placeholder="Ask a question about your documents…"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKey}
        disabled={isRunning}
      />
      <div className="chat-input-actions">
        <button
          className="chat-clear-btn"
          onClick={clearHistory}
          title="Clear conversation"
          disabled={isRunning}
        >
          ✕
        </button>
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={isRunning || !text.trim()}
        >
          {isRunning ? (
            <span className="send-spinner" />
          ) : (
            "↑"
          )}
        </button>
      </div>
    </div>
  );
}
