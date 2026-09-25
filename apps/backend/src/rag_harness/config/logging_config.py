"""
Single shared logging setup, called once at process start (CLI scripts,
API startup) — avoids each module configuring its own handlers.

Uses `rich` for colored, readable log output in the VS Code terminal:
level-colored tags, dimmed logger names, and a clean traceback format on
uncaught exceptions. Falls back to plain logging if `rich` isn't installed
(e.g. minimal CI environments) so this never breaks a run.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger("rag_harness")
    if root_logger.handlers:
        return  # already configured, avoid duplicate handlers on re-import

    root_logger.setLevel(level)

    try:
        from rich.console import Console
        from rich.logging import RichHandler

        handler = RichHandler(
            console=Console(stderr=False),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            log_time_format="[%H:%M:%S]",
        )
        formatter = logging.Formatter("%(name)s | %(message)s")
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Noisy third-party libraries — keep them at WARNING so the harness's
    # own INFO logs (the "story" of a query) aren't buried.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
