// =============================================================================
// FILE: src/App.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/App.tsx
// REPLACES: existing App.tsx (full replacement)
//
// WHAT CHANGED:
//   The 3-column zone layout (Knowledge | QueryWorkspace | Config) is replaced
//   with a 3-tab layout: Chat | Ingest | Config
//   The Debug Panel is now a floating overlay toggled from the Topbar.
//   The old zone-based workspace grid is gone.
//
// WHY:
//   Notes specify "a chat application UI (good looking)" as the main UX.
//   Tabs are the right pattern for a tool with 3 distinct modes.
//   Debug Panel as overlay lets the user inspect agent internals without
//   collapsing the chat.
//
// WHERE IT CONNECTS:
//   components/layout/Topbar.tsx (tab nav + debug toggle)
//   components/chat/ChatPage.tsx (Chat tab)
//   components/ingest/IngestPage.tsx (Ingest tab)
//   components/config/ConfigPage.tsx (Config tab)
//   components/debug/DebugPanel.tsx (overlay)
// =============================================================================

import { useState } from "react";
import { Topbar } from "@/components/layout/Topbar";
import { ChatPage } from "@/components/chat/ChatPage";
import { IngestPage } from "@/components/ingest/IngestPage";
import { ConfigPage } from "@/components/config/ConfigPage";
import { DebugPanel } from "@/components/debug/DebugPanel";

export type AppTab = "chat" | "ingest" | "config";

export default function App() {
  const [tab, setTab] = useState<AppTab>("chat");

  return (
    <div className="app-shell">
      <Topbar activeTab={tab} onTabChange={setTab} />
      <div className="app-body">
        {tab === "chat" && <ChatPage />}
        {tab === "ingest" && <IngestPage />}
        {tab === "config" && <ConfigPage />}
      </div>
      <DebugPanel />
    </div>
  );
}
