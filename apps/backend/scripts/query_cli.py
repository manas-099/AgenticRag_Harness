"""
CLI entry point for querying the RAG harness.
Usage:
  python scripts/query_cli.py --question "What is the LCR?"
  python scripts/query_cli.py --question "What is the LCR?" --agentic
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

from rag_harness.api.dependencies import get_rag_pipeline  # noqa: E402
from rag_harness.config.logging_config import setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Query the RAG harness")
    parser.add_argument("--question", required=True, help="The question to ask")
    parser.add_argument("--agentic", action="store_true", help="Use the full ReAct agent loop instead of a single retrieve+generate pass")
    parser.add_argument("--doc-version-key", default="v1", help="Cache key for document version (default: v1)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("rag_harness.cli.query")

    pipeline = get_rag_pipeline()

    if pipeline.is_empty():
        print("No documents have been ingested yet. Run ingest_cli.py first.")
        return

    if args.agentic:
        result = pipeline.answer_agentic(args.question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources used: {result['sources_used']}")
        print(f"Degraded: {result['degraded']} (reason: {result.get('degrade_reason')})")
        print(f"Action history: {result.get('action_history', [])}")
    else:
        result = pipeline.answer(args.question, doc_version_key=args.doc_version_key)
        print(f"\nAnswer: {result.answer}")
        print(f"Sources used: {result.sources_used}")
        print(f"Insufficient info: {result.is_insufficient}")
        print(f"From cache: {result.from_cache}")

    logger.info("Query complete")


if __name__ == "__main__":
    main()