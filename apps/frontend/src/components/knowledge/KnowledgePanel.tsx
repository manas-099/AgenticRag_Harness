/**
 * KnowledgePanel.tsx — Zone A: document count / chunk count stat cards,
 * the drop zone, and the ingested-document list with status badges.
 */
import { useKnowledgeStore } from "@/store/knowledgeStore";
import { DropZone } from "@/components/knowledge/DropZone";

export function KnowledgePanel() {
  const documents = useKnowledgeStore((s) => s.documents);
  const totalChunks = useKnowledgeStore((s) => s.totalChunks);

  return (
    <div className="zone-body">
      <div className="kb-stats">
        <div className="stat-card">
          <div className="num">{documents.length}</div>
          <div className="lbl">documents</div>
        </div>
        <div className="stat-card">
          <div className="num">{totalChunks}</div>
          <div className="lbl">chunks</div>
        </div>
      </div>

      <DropZone />

      {documents.length > 0 && <div className="doc-section-label">Documents</div>}
      {documents.map((doc) => (
        <div className="doc-card" key={doc.docId}>
          <div className="doc-card-top">
            <span className="doc-card-name">{doc.fileName}</span>
            <span className={`badge ${doc.status}`}>{doc.status}</span>
          </div>
          <div className="doc-card-meta">
            {doc.status === "error"
              ? doc.errorMessage
              : `${doc.chunkCount} chunks · ${doc.version}`}
          </div>
        </div>
      ))}
    </div>
  );
}
