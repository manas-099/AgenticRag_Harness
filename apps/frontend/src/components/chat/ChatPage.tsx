// =============================================================================
// FILE: src/components/chat/ChatPage.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/chat/ChatPage.tsx
// NEW FILE
//
// WHAT IT DOES:
//   The Chat tab layout. Two columns:
//     Left (flex 1): ChatWindow (message bubbles) + ChatInput (bottom bar)
//     Right (280px, slides in when agent has steps): StepPanel
//
//   The right panel is visible only when steps[] is non-empty (i.e. an
//   agentic run is in progress or just finished). It shows the ReAct step
//   graph + the detailed step list.
//
//   Mode toggle (Standard / Agentic) lives above the chat window.
//
// WHY:
//   Notes say: "Chat UI — user → chat UI → steps panel (ingest → retrieval
//   flow shown)" and "when agentic RAG is running → the idle UI should show
//   something is doing work like the thinking animation in Claude".
//
// WHERE IT CONNECTS:
//   store/chatStore.ts (mode, steps, isRunning)
//   components/chat/ChatWindow.tsx
//   components/chat/ChatInput.tsx
//   components/chat/ModeToggle.tsx
//   components/steps/StepPanel.tsx
// =============================================================================

import { useChatStore } from "@/store/chatStore";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { ModeToggle } from "@/components/chat/ModeToggle";
import { StepPanel } from "@/components/steps/StepPanel";

export function ChatPage() {
  const steps = useChatStore((s) => s.steps);
  const showSteps = steps.length > 0;

  return (
    <div className="chat-page">
      <div className="chat-main">
        <div className="chat-topbar">
          <ModeToggle />
        </div>
        <ChatWindow />
        <ChatInput />
      </div>
      {showSteps && (
        <aside className="step-panel-aside">
          <StepPanel />
        </aside>
      )}
    </div>
  );
}
