// =============================================================================
// FILE: src/utils/stripCitations.ts
// TARGET PATH: apps/frontend/src/utils/stripCitations.ts
// NEW FILE
//
// WHAT IT DOES:
//   Strips inline citations like [Doc: ticket_tk..., Page: 0, chunk_id: 712...]
//   from the LLM answer text before rendering in the chat bubble.
//   Citations should only appear in the Debug Panel, not in the user-facing answer.
//
// WHY:
//   The RAG_SYSTEM_PROMPT instructs the LLM to inline citations in every
//   sentence: [Doc: <doc_id>, Page: <page_num>, chunk_id: <chunk_id>]
//   This is correct for verification but clutters the user-facing answer.
//   We strip them here in the frontend — backend unchanged.
//
// USAGE:
//   import { stripCitations } from "@/utils/stripCitations";
//   const clean = stripCitations(rawAnswer);
// =============================================================================

/**
 * Removes [Doc: ..., Page: ..., chunk_id: ...] citations from answer text.
 * Also trims any double-spaces or orphaned periods left behind.
 */
export function stripCitations(text: string): string {
  return text
    // Remove full citation blocks: [Doc: ..., Page: ..., chunk_id: ...]
    .replace(/\s*\[Doc:[^\]]+\]/g, "")
    // Clean up any double spaces left behind
    .replace(/  +/g, " ")
    // Clean up " ." or " ," artifacts
    .replace(/ ([.,;])/g, "$1")
    .trim();
}
