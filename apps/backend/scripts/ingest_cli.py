"""
CLI entry point for ingesting a document.
Usage: python scripts/ingest_cli.py --file path/to/doc.pdf --doc-id my_doc --version v1
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

from rag_harness.api.dependencies import get_rag_pipeline  # noqa: E402
from rag_harness.config.logging_config import setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG harness")
    parser.add_argument("--file", required=True, help="Path to PDF or MD file")
    parser.add_argument("--doc-id", required=True, help="Document identifier")
    parser.add_argument("--version", default="v1", help="Document version (default: v1)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("rag_harness.cli.ingest")

    pipeline = get_rag_pipeline()
    chunks = pipeline.ingest_document(args.file, args.doc_id, args.version)

    logger.info(f"Ingested {len(chunks)} chunks from {args.file}")
    print(f"\nDone. {len(chunks)} chunks stored for doc_id={args.doc_id}, version={args.version}")
    print(f"Pipeline status: {pipeline.status()}")


if __name__ == "__main__":
    main()