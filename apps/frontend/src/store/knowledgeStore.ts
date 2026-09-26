/**
 * knowledgeStore.ts
 * ---------------------------------------------------------------------------
 * Tracks documents ingested through this UI. The backend has no
 * "list ingested documents" endpoint yet, so this is the client's own
 * record — accurate for anything ingested via this app in the current
 * browser, cleared on refresh. See README "Known gaps" for the small
 * `GET /v1/documents` addition that would make this authoritative/persisted
 * instead.
 */
import { create } from "zustand";
import { ingestDocument } from "@/api/documentsApi";
import type { KnowledgeDocument } from "@/types/domain";

interface KnowledgeStore {
  documents: KnowledgeDocument[];
  totalChunks: number;
  ingest: (fileName: string) => Promise<void>;
}

function slugify(fileName: string): string {
  return fileName
    .replace(/\.[^/.]+$/, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .slice(0, 48);
}

export const useKnowledgeStore = create<KnowledgeStore>()((set) => ({
  documents: [],
  totalChunks: 0,

  ingest: async (fileName: string) => {
    const docId = slugify(fileName) || `doc_${Date.now()}`;
    const placeholder: KnowledgeDocument = {
      docId,
      fileName,
      version: "v1",
      chunkCount: 0,
      status: "ingesting",
    };
    set((s) => ({ documents: [placeholder, ...s.documents] }));

    try {
      const result = await ingestDocument(fileName, docId, "v1");
      set((s) => ({
        documents: s.documents.map((d) =>
          d.docId === docId ? { ...d, status: "ready", chunkCount: result.chunks_created } : d,
        ),
        totalChunks: s.totalChunks + result.chunks_created,
      }));
    } catch (err) {
      set((s) => ({
        documents: s.documents.map((d) =>
          d.docId === docId
            ? { ...d, status: "error", errorMessage: err instanceof Error ? err.message : "Ingestion failed" }
            : d,
        ),
      }));
    }
  },
}));

export const selectDocumentCount = (s: KnowledgeStore) => s.documents.length;
