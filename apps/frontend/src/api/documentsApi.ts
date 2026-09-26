/**
 * documentsApi.ts — wrapper around POST /v1/documents.
 *
 * KNOWN GAP: the current backend endpoint ingests a file already sitting on
 * the SERVER's filesystem (`file_path`), not a browser-uploaded file — there
 * is no multipart upload route yet. Real drag-and-drop-from-browser ingest
 * needs a small backend addition:
 *
 *   @router.post("/documents/upload")
 *   async def upload_document(file: UploadFile, doc_id: str = Form(...)):
 *       tmp_path = save_to_disk(file)
 *       return ingest_use_case.execute(tmp_path, doc_id, "v1")
 *
 * Until that exists, this function sends the file's name as `file_path` —
 * it will only actually ingest something if a file of that name already
 * exists on the server (e.g. you've placed sample PDFs there for a demo).
 * The Knowledge panel is built to swap this over transparently once the
 * upload endpoint exists — see components/knowledge/DropZone.tsx.
 */
import { apiRequest } from "@/api/client";
import type { IngestDocumentResult } from "@/types/domain";

export function ingestDocument(fileName: string, docId: string, docVersion = "v1"): Promise<IngestDocumentResult> {
  return apiRequest<IngestDocumentResult>("/documents", {
    method: "POST",
    body: { file_path: fileName, doc_id: docId, doc_version: docVersion },
  });
}
