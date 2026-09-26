/**
 * App.tsx — top-level shell: Topbar + 3-column workspace
 * (Knowledge | Query workspace | Config), with collapsible side columns.
 */
import { useState } from "react";
import { Topbar } from "@/components/layout/Topbar";
import { CollapsibleZone } from "@/components/layout/CollapsibleZone";
import { KnowledgePanel } from "@/components/knowledge/KnowledgePanel";
import { QueryWorkspace } from "@/components/query/QueryWorkspace";
import { ConfigPanel } from "@/components/config/ConfigPanel";

export default function App() {
  const [aCollapsed, setACollapsed] = useState(false);
  const [cCollapsed, setCCollapsed] = useState(false);

  return (
    <div className="shell">
      <Topbar />
      <div className={`workspace ${aCollapsed ? "a-collapsed" : ""} ${cCollapsed ? "c-collapsed" : ""}`}>
        <CollapsibleZone name="Knowledge" zoneClass="zone-a" onCollapsedChange={setACollapsed}>
          <KnowledgePanel />
        </CollapsibleZone>

        <QueryWorkspace />

        <CollapsibleZone name="Config" zoneClass="zone-c" onCollapsedChange={setCCollapsed}>
          <ConfigPanel />
        </CollapsibleZone>
      </div>
    </div>
  );
}
