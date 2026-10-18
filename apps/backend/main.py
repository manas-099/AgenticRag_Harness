# ── Eval structured logger ────────────────────────────────────────────────────
# Writes one JSON line per agentic query to evals/logs/eval_events.jsonl.
# Safe to leave on in production — it only appends, never blocks, never raises.
import logging
import os

_eval_log_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "evals", "logs", "eval_events.jsonl"
)
_eval_file_handler = logging.FileHandler(_eval_log_path, mode="a", encoding="utf-8")
_eval_file_handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON lines only

eval_logger = logging.getLogger("rag_harness.eval")
eval_logger.setLevel(logging.INFO)
eval_logger.addHandler(_eval_file_handler)
eval_logger.propagate = False  # do not duplicate into root logger / uvicorn output
# ─────────────────────────────────────────────────────────────────────────────


from rag_harness.api.main import app  # noqa: F401


def main():
    print("Hello from backend!")


if __name__ == "__main__":
    main()

