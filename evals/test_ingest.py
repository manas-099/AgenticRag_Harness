"""
Auto-ingests both NBFC corpus PDFs into the running backend.
Run once after starting the server: python test_ingest.py
"""
import requests
import os
import sys

BASE_URL = "http://localhost:8000/api/v1"

DOCS = [
    {
        "path": "docs/corpus/nbfc_responsible_business_conduct_2025.pdf",
        "doc_id": "nbfc_rbc_2025",
        "doc_title": "NBFC Responsible Business Conduct Directions 2025",
    },
    {
        "path": "docs/corpus/nbfc_scale_based_regulation_2023.pdf",
        "doc_id": "nbfc_sbr_2023",
        "doc_title": "NBFC Scale Based Regulation Directions 2023",
    },
]


def ingest(doc: dict) -> None:
    path = doc["path"]
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        return

    print(f"[INGEST] {doc['doc_id']} ← {path}")
    with open(path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/ingest",
            files={"file": (os.path.basename(path), f, "application/pdf")},
            data={"doc_id": doc["doc_id"], "doc_title": doc["doc_title"]},
            timeout=120,
        )

    if resp.status_code == 200:
        print(f"  ✓ Done — {resp.json()}")
    else:
        print(f"  ✗ Failed ({resp.status_code}): {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    for doc in DOCS:
        ingest(doc)
    print("\nAll docs ingested.")