/**
 * documentsApi.ts — wrapper around POST /v1/documents/upload.
 *
 * Sends the actual File object as multipart/form-data so the backend can
 * save it to a temp directory, ingest it with PyMuPDF/Docling, and clean up.
 * The previous stub that only sent `file_path: fileName` is replaced here;
 * the old endpoint (POST /v1/documents with a JSON body) still exists on the
 * backend for server-side scripting use cases.
 */
import { apiRequest } from "@/api/client";
import type { IngestDocumentResult } from "@/types/domain";

export function ingestDocument(file: File, docId: string, docVersion = "v1"): Promise<IngestDocumentResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("doc_id", docId);
  form.append("doc_version", docVersion);

  return apiRequest<IngestDocumentResult>("/documents/upload", {
    method: "POST",
    // Pass FormData directly — apiRequest must NOT set Content-Type so
    // the browser can set the correct multipart boundary automatically.
    body: form,
  });
}
