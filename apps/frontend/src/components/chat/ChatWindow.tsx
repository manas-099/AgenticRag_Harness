// =============================================================================
// FILE: src/components/chat/ChatWindow.tsx
// TARGET PATH: apps/frontend/src/components/chat/ChatWindow.tsx
// REPLACES: previous ChatWindow.tsx
//
// WHAT CHANGED:
//   - Citations stripped via stripCitations() (done in chatStore, text is clean here)
//   - Cache hit badge: when from_cache=true, shows a ⚡ cache hit chip
//   - DegradedCard with human-readable reason (kept from v2)
//   - Good answer chips: sources (teal), iter, cache hit (amber)
// =============================================================================

import { useEffect, useRef } from "react";
import { useChatStore } from "@/store/chatStore";
import type { ChatMessage } from "@/types/domain";

const DEGRADE_MESSAGES: Record<string, { title: string; hint: string }> = {
  max_iterations_exceeded: {
    title: "Couldn't find a confident answer in time",
    hint: "Try a more specific question, or check the relevant document is ingested.",
  },
  token_budget_exceeded: {
    title: "Question required too much searching",
    hint: "Try breaking your question into smaller parts.",
  },
  max_search_attempts_exceeded: {
    title: "Searched extensively but found no matching content",
    hint: "The answer may not be in your documents. Try rephrasing or uploading a relevant file.",
  },
  stuck_loop_no_new_chunks: {
    title: "Agent got stuck — repeated searches found no new information",
    hint: "Try rephrasing with different keywords, or add more documents.",
  },
  max_validation_retries_exceeded: {
    title: "Found something but couldn't verify it confidently",
    hint: "The answer may exist but the agent couldn't confirm it. Try rephrasing.",
  },
  no_chunks_retrieved: {
    title: "No relevant content found in your documents",
    hint: "Make sure you've ingested a document containing the answer.",
  },
  insufficient_context: {
    title: "Not enough information in documents",
    hint: "Try uploading more relevant documents.",
  },
};

const DEFAULT_DEGRADE = {
  title: "Couldn't produce a verified answer",
  hint: "Try rephrasing, or open the Debug Panel (⬡) for details.",
};

function ThinkingDots() {
  return (
    <div className="thinking-dots">
      <span /><span /><span />
    </div>
  );
}

function DegradedCard({ reason }: { reason: string | null }) {
  const info = (reason ? DEGRADE_MESSAGES[reason] : undefined) || DEFAULT_DEGRADE;
  return (
    <div className="degraded-card">
      <div className="degraded-card-icon">⚠</div>
      <div className="degraded-card-body">
        <div className="degraded-card-title">{info.title}</div>
        <div className="degraded-card-hint">{info.hint}</div>
        <div className="degraded-card-tips">
          <span>• Rephrase the question</span>
          <span>• Check the Ingest tab — is the right document uploaded?</span>
          <span>• Open the Debug Panel (⬡) to see what the agent tried</span>
        </div>
      </div>
    </div>
  );
}

function MessageMeta({ msg }: { msg: ChatMessage }) {
  const meta = msg.agenticMeta;
  if (!meta || meta.degraded) return null;
  return (
    <div className="msg-meta">
      {meta.from_cache && (
        <span className="msg-chip cache-hit" title="Served from cache — no agent run needed">
          ⚡ cache hit
        </span>
      )}
      {meta.sources_used.length > 0 && (
        <span className="msg-chip teal">{meta.sources_used.length} sources</span>
      )}
      {!meta.from_cache && meta.iterations_used > 0 && (
        <span className="msg-chip">{meta.iterations_used} iter</span>
      )}
      {meta.validation_retries > 0 && (
        <span className="msg-chip">{meta.validation_retries} verify loops</span>
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
      {messages.map((msg) => {
        const isDegraded = msg.agenticMeta?.degraded === true;
        return (
          <div key={msg.id} className={`chat-row ${msg.role}`}>
            {msg.role === "assistant" && <div className="avatar">H</div>}
            <div className={`chat-bubble ${msg.role} ${isDegraded ? "degraded" : ""}`}>
              {msg.isLoading ? (
                <ThinkingDots />
              ) : isDegraded ? (
                <DegradedCard reason={msg.agenticMeta?.degrade_reason ?? null} />
              ) : (
                <>
                  <div className="bubble-text">{msg.content}</div>
                  <MessageMeta msg={msg} />
                </>
              )}
            </div>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
