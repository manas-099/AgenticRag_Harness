// =============================================================================
// FILE: src/components/ingest/IngestPage.tsx
// TARGET PATH IN PROJECT: apps/frontend/src/components/ingest/IngestPage.tsx
// NEW FILE (replaces KnowledgePanel in the old zone-A)
//
// WHAT IT DOES:
//   Full-page ingest tab. Sections:
//
//   1. STAT BAR — doc count, total chunks, chunk avg size (chunks/docs),
//      qdrant count from health
//   2. DROP ZONE — drag-and-drop or click to upload .pdf / .md files.
//      Shows per-file progress inline (ingesting → ready / error badge).
//      Accepts multiple files at once; each starts ingesting immediately.
//   3. DOCUMENT LIST — table-like list with columns:
//      Name | ID (doc_id) | Chunks | Status
//      Notes say "once ingested save in the UI with ID and name" — this is it.
//   4. CACHE STATUS CARDS — live from GET /v1/health:
//      "Query cache: N entries" + "Retrieval cache: N entries"
//      Notes say "Show query cached or not, retrieval cached or not — the 2
//      caching layers"
//
// WHY:
//   Notes say: "I can add many docs also, once injected save in the UI with
//   ID and name. Show chunk size, chunk amount."
//   Also: "Show query cached or not, retrieval cached or not."
//
// WHERE IT CONNECTS:
//   store/knowledgeStore.ts (ingest, documents, totalChunks)
//   api/healthApi.ts (fetchBackendHealth for cache stats)
//   components/knowledge/DropZone.tsx (reused drag-drop component)
// =============================================================================

import { useEffect, useState } from "react";
import { useKnowledgeStore } from "@/store/knowledgeStore";
import { fetchBackendHealth } from "@/api/healthApi";
import type { BackendHealth } from "@/types/domain";

function DropZone() {
  const ingest = useKnowledgeStore((s) => s.ingest);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files: FileList | null) {
    if (!files) return;
    Array.from(files).forEach((f) => ingest(f));
  }

  return (
    <div
      className={`drop-zone ${dragging ? "drag-over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      onClick={() => document.getElementById("file-input")?.click()}
    >
      <span className="plus">+</span>
      Drop .pdf or .md files here, or click to browse
      <input
        id="file-input"
        type="file"
        accept=".pdf,.md"
        multiple
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}

export function IngestPage() {
  const { documents, totalChunks } = useKnowledgeStore();
  const [health, setHealth] = useState<BackendHealth | null>(null);

  useEffect(() => {
    fetchBackendHealth().then(setHealth).catch(() => {});
    const t = setInterval(() => fetchBackendHealth().then(setHealth).catch(() => {}), 15_000);
    return () => clearInterval(t);
  }, []);

  const avgChunks = documents.length > 0
    ? Math.round(totalChunks / documents.filter((d) => d.status === "ready").length || 0)
    : 0;

  return (
    <div className="ingest-page">
      {/* Stat bar */}
      <div className="ingest-stat-bar">
        <div className="istat">
          <div className="istat-val">{documents.length}</div>
          <div className="istat-lbl">documents</div>
        </div>
        <div className="istat">
          <div className="istat-val">{totalChunks}</div>
          <div className="istat-lbl">total chunks</div>
        </div>
        <div className="istat">
          <div className="istat-val">{avgChunks}</div>
          <div className="istat-lbl">avg chunks/doc</div>
        </div>
        {health && (
          <div className="istat">
            <div className="istat-val">{health.qdrant_chunk_count}</div>
            <div className="istat-lbl">in Qdrant</div>
          </div>
        )}
      </div>

      {/* Drop zone */}
      <div className="ingest-section">
        <div className="ingest-section-label">Add Documents</div>
        <DropZone />
      </div>

      {/* Cache status cards */}
      {health && (
        <div className="ingest-section">
          <div className="ingest-section-label">Cache Status</div>
          <div className="cache-cards">
            <div className="cache-card">
              <div className="cache-card-label">Query Cache</div>
              <div className={`cache-card-val ${health.query_cache_size > 0 ? "hit" : ""}`}>
                {health.query_cache_size} entries
              </div>
              <div className="cache-card-hint">
                {health.query_cache_size > 0 ? "● cached results available" : "○ no cached results yet"}
              </div>
            </div>
            <div className="cache-card">
              <div className="cache-card-label">Retrieval Cache</div>
              <div className={`cache-card-val ${health.retrieval_cache_size > 0 ? "hit" : ""}`}>
                {health.retrieval_cache_size} entries
              </div>
              <div className="cache-card-hint">
                {health.retrieval_cache_size > 0 ? "● cached chunks available" : "○ no cached chunks yet"}
              </div>
            </div>
            <div className="cache-card">
              <div className="cache-card-label">BM25 Index</div>
              <div className={`cache-card-val ${health.sparse_index_built ? "hit" : ""}`}>
                {health.sparse_corpus_size} docs
              </div>
              <div className="cache-card-hint">
                {health.sparse_index_built ? "● index built" : "○ not yet built"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Document list */}
      {documents.length > 0 && (
        <div className="ingest-section">
          <div className="ingest-section-label">Documents</div>
          <div className="doc-table">
            <div className="doc-table-head">
              <span>Name</span>
              <span>Doc ID</span>
              <span>Chunks</span>
              <span>Status</span>
            </div>
            {documents.map((doc) => (
              <div className="doc-table-row" key={doc.docId}>
                <span className="doc-name">{doc.fileName}</span>
                <span className="doc-id-cell">{doc.docId}</span>
                <span className="doc-chunks">{doc.chunkCount || "—"}</span>
                <span className={`badge ${doc.status}`}>{doc.status}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
