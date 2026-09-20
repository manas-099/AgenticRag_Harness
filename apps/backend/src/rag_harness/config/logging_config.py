"""
Single shared logging setup, called once at process start (CLI scripts,
API startup) — avoids each module configuring its own handlers.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger("rag_harness")
    if root_logger.handlers:
        return  # already configured, avoid duplicate handlers on re-import

    root_logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)