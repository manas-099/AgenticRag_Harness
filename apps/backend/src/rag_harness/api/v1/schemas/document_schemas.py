from pydantic import BaseModel


class IngestDocumentRequest(BaseModel):
    file_path: str
    doc_id: str
    doc_version: str = "v1"


class IngestDocumentResponse(BaseModel):
    doc_id: str
    chunks_created: int