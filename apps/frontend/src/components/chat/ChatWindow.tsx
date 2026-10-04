// =============================================================================
// FILE: src/components/chat/ChatWindow.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/chat/ChatWindow.tsx
// NEW FILE
//
// WHAT IT DOES:
//   Renders the scrollable list of chat messages.
//   User messages: right-aligned dark bubble.
//   Assistant messages: left-aligned surface bubble.
//   Loading state: shows a 3-dot "thinking" animation (like Claude's UI)
//   inside the assistant bubble while isLoading=true.
//   After a response: shows the answer text + metadata chips (source count,
//   iterations used, degraded warning if applicable).
//   Auto-scrolls to bottom on new message.
//
// WHY:
//   Notes say "when agentic RAG is running the idle UI should show something
//   doing work like thinking animation in Claude". The 3-dot pulse is that.
//
// WHERE IT CONNECTS:
//   store/chatStore.ts (messages[])
//   styles/app.css (.chat-bubble, .thinking-dots, .msg-meta)
// =============================================================================

import { useEffect, useRef } from "react";
import { useChatStore } from "@/store/chatStore";
import type { ChatMessage } from "@/types/domain";

function ThinkingDots() {
  return (
    <div className="thinking-dots">
      <span />
      <span />
      <span />
    </div>
  );
}

function MessageMeta({ msg }: { msg: ChatMessage }) {
  const meta = msg.agenticMeta;
  if (!meta) return null;
  return (
    <div className="msg-meta">
      {meta.degraded && (
        <span className="msg-chip degraded" title={meta.degrade_reason ?? ""}>
          ⚠ degraded
        </span>
      )}
      {meta.iterations_used > 0 && (
        <span className="msg-chip">{meta.iterations_used} iter</span>
      )}
      {meta.sources_used.length > 0 && (
        <span className="msg-chip">{meta.sources_used.length} sources</span>
      )}
      {meta.validation_retries > 0 && (
        <span className="msg-chip">{meta.validation_retries} retries</span>
      )}
    </div>
  );
}

export function ChatWindow() {
  const messages = useChatStore((s) => s.messages);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="chat-window empty-state">
        <div className="empty-icon">⬡</div>
        <div className="empty-title">Agentic RAG Harness</div>
        <div className="empty-sub">
          Ask a question — the agent will search, verify and synthesize an answer from your documents.
        </div>
      </div>
    );
  }

  return (
    <div className="chat-window">
      {messages.map((msg) => (
        <div key={msg.id} className={`chat-row ${msg.role}`}>
          {msg.role === "assistant" && (
            <div className="avatar">H</div>
          )}
          <div className={`chat-bubble ${msg.role}`}>
            {msg.isLoading ? (
              <ThinkingDots />
            ) : (
              <>
                <div className="bubble-text">{msg.content}</div>
                <MessageMeta msg={msg} />
              </>
            )}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
