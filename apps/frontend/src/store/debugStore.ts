// =============================================================================
// FILE: src/store/debugStore.ts
// TARGET PATH IN PROJECT: apps/frontend/src/store/debugStore.ts
// NEW FILE (does not exist yet)
//
// WHAT IT DOES:
//   Lightweight store that holds the debug snapshot of the last agentic run:
//     - query, degraded, degrade_reason, iterations_used, validation_retries
//     - token_budget_used (and max from health endpoint for % bar)
//     - action_steps with thought text per step
//     - isOpen: whether the Debug Panel overlay is visible
//
//   setLastResult() is called by chatStore.send() after each agentic response.
//
// WHY:
//   Keeps debug state separate from chat state so DebugPanel can be opened/
//   closed independently and doesn't cause the chat to re-render.
//
// WHERE IT CONNECTS:
//   store/chatStore.ts calls setLastResult()
//   components/debug/DebugPanel.tsx reads this store
//   components/layout/Topbar.tsx toggles isOpen
// =============================================================================

import { create } from "zustand";
import type { DebugState } from "@/types/domain";

interface DebugStore {
  lastResult: DebugState | null;
  isOpen: boolean;
  setLastResult: (result: DebugState) => void;
  toggleOpen: () => void;
  setOpen: (open: boolean) => void;
}

export const useDebugStore = create<DebugStore>()((set) => ({
  lastResult: null,
  isOpen: false,
  setLastResult: (result) => set({ lastResult: result }),
  toggleOpen: () => set((s) => ({ isOpen: !s.isOpen })),
  setOpen: (open) => set({ isOpen: open }),
}));
