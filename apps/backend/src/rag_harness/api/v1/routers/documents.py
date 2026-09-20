from fastapi import APIRouter, Depends, HTTPException

from rag_harness.api.dependencies import get_rag_pipeline
from rag_harness.api.v1.schemas import IngestDocumentRequest, IngestDocumentResponse
from rag_harness.application.rag_pipeline import RAGPipeline
from rag_harness.domain.exceptions import HarnessError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestDocumentResponse)
def ingest_document(request: IngestDocumentRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    try:
        chunks = pipeline.ingest_document(request.file_path, request.doc_id, request.doc_version)
        return IngestDocumentResponse(doc_id=request.doc_id, chunks_created=len(chunks))
    except HarnessError as e:
        raise HTTPException(status_code=422, detail=str(e))