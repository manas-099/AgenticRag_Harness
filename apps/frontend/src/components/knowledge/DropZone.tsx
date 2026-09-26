/**
 * DropZone.tsx — drag-and-drop (or click-to-browse) file target that kicks
 * off ingestion via useKnowledgeStore. See api/documentsApi.ts for the
 * "server file_path vs browser upload" gap this currently works around.
 */
import { useRef, useState } from "react";
import { useKnowledgeStore } from "@/store/knowledgeStore";

export function DropZone() {
  const ingest = useKnowledgeStore((s) => s.ingest);
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    if (!files) return;
    Array.from(files).forEach((file) => void ingest(file));
  }


  return (
    <>
      <button
        type="button"
        className={`drop-zone ${isDragOver ? "drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <span className="plus">+</span>
        Drop a file to ingest
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        hidden
        accept=".pdf,.md,.txt,.docx"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </>
  );
}
