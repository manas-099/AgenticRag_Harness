import shutil
import tempfile
import os

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from rag_harness.api.dependencies import get_rag_pipeline
from rag_harness.api.v1.schemas import IngestDocumentRequest, IngestDocumentResponse
from rag_harness.application.rag_pipeline import RAGPipeline
from rag_harness.domain.exceptions import HarnessError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestDocumentResponse)
def ingest_document(request: IngestDocumentRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """Ingest a file that already exists on the server's filesystem."""
    try:
        chunks = pipeline.ingest_document(request.file_path, request.doc_id, request.doc_version)
        return IngestDocumentResponse(doc_id=request.doc_id, chunks_created=len(chunks))
    except HarnessError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/upload", response_model=IngestDocumentResponse)
async def upload_and_ingest(
    file: UploadFile,
    doc_id: str = Form(...),
    doc_version: str = Form("v1"),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Accept a browser file upload, persist it temporarily, ingest, then clean up."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in (".pdf", ".md"):
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix}'. Only .pdf and .md are accepted.")

    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, file.filename or f"upload{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = pipeline.ingest_document(tmp_path, doc_id, doc_version)
        return IngestDocumentResponse(doc_id=doc_id, chunks_created=len(chunks))
    except HarnessError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)