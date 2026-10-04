// =============================================================================
// FILE: src/App.tsx
// TARGET PATH: apps/frontend/src/App.tsx
// REPLACES: previous App.tsx
//
// WHAT CHANGED:
//   Layout now has a LEFT SIDEBAR for Config (collapsible to 44px icon rail)
//   instead of Config being a top tab. Tab bar now only has: Ingest | Chat
//   The sidebar shows: Retrieval settings, Agent settings, Model settings.
//   Book+animals SVG in the topbar next to "RAG Harness" brand name.
// =============================================================================

import { useState } from "react";
import { Topbar } from "@/components/layout/Topbar";
import { ConfigSidebar } from "@/components/config/ConfigSidebar";
import { ChatPage } from "@/components/chat/ChatPage";
import { IngestPage } from "@/components/ingest/IngestPage";
import { DebugPanel } from "@/components/debug/DebugPanel";

export type AppTab = "chat" | "ingest";

export default function App() {
  const [tab, setTab] = useState<AppTab>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="app-shell">
      <Topbar activeTab={tab} onTabChange={setTab} />
      <div className="app-body">
        <ConfigSidebar open={sidebarOpen} onToggle={() => setSidebarOpen((v) => !v)} />
        <div className="app-main">
          {tab === "chat" && <ChatPage />}
          {tab === "ingest" && <IngestPage />}
        </div>
      </div>
      <DebugPanel />
    </div>
  );
}
