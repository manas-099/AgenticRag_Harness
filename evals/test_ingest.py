"""
Auto-ingests both NBFC corpus PDFs into the running backend.
Run once after starting the server: python test_ingest.py

NOTE: First run downloads Docling OCR models. Allow up to 10 minutes.
"""
# pyrefly: ignore[missing-import]
import requests
import os
import sys
import time

BASE_URL = "http://localhost:8000/api/v1"
INGEST_TIMEOUT = 600  # seconds — Docling model download + OCR + embedding is slow on first run

DOCS = [
    {
        "path": "docs/corpus/NBFC Responsible Business Conduct Directions-1-10.pdf",
        "doc_id": "nbfc_rbc_2025",
        "doc_title": "NBFC Responsible Business Conduct Directions 2025",
    },
    {
        "path": "docs/corpus/NBFC_rulebook-1-10.pdf",
        "doc_id": "nbfc_sbr_2023",
        "doc_title": "NBFC Scale Based Regulation Directions 2023",
    },
]


def ingest(doc: dict) -> None:
    path = doc["path"]
    if not os.path.exists(path):
        # If running from evals/ directory, check relative to repo root
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"[SKIP] File not found: {path}")
            return

    print(f"[INGEST] {doc['doc_id']} ← {path}")
    print(f"  ⏳ Sending to server (timeout={INGEST_TIMEOUT}s — first run may be slow)...")
    t0 = time.time()
    with open(path, "rb") as f:
        # Try /documents/upload first, fallback to /ingest
        upload_url = f"{BASE_URL}/documents/upload"
        resp = requests.post(
            upload_url,
            files={"file": (os.path.basename(path), f, "application/pdf")},
            data={"doc_id": doc["doc_id"], "doc_version": "v1"},
            timeout=INGEST_TIMEOUT,
        )
        if resp.status_code == 404:
            f.seek(0)
            resp = requests.post(
                f"{BASE_URL}/ingest",
                files={"file": (os.path.basename(path), f, "application/pdf")},
                data={"doc_id": doc["doc_id"], "doc_title": doc["doc_title"]},
                timeout=INGEST_TIMEOUT,
            )

    elapsed = time.time() - t0
    if resp.status_code == 200:
        print(f"  ✓ Done in {elapsed:.0f}s — {resp.json()}")
    else:
        print(f"  ✗ Failed in {elapsed:.0f}s ({resp.status_code}): {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    for doc in DOCS:
        ingest(doc)
    print("\nAll docs ingested.")