/**
 * CollapsibleZone.tsx — generic wrapper for Zone A / Zone C: a header with a
 * name + collapse button, collapsing to a 44px rail (see .workspace.a-collapsed
 * / .c-collapsed in app.css) while keeping a small dot rail visible.
 */
import { useState, type ReactNode } from "react";

interface CollapsibleZoneProps {
  name: string;
  zoneClass: string;
  onCollapsedChange?: (collapsed: boolean) => void;
  children: ReactNode;
}

export function CollapsibleZone({ name, zoneClass, onCollapsedChange, children }: CollapsibleZoneProps) {
  const [collapsed, setCollapsed] = useState(false);

  function toggle() {
    const next = !collapsed;
    setCollapsed(next);
    onCollapsedChange?.(next);
  }

  return (
    <div className={`zone ${zoneClass} ${collapsed ? "collapsed" : ""}`}>
      <div className="zone-head">
        <span className="zone-name">{name}</span>
        <button className="collapse-btn" onClick={toggle} aria-label={collapsed ? "Expand" : "Collapse"}>
          {collapsed ? ">" : "<"}
        </button>
      </div>
      {children}
    </div>
  );
}
