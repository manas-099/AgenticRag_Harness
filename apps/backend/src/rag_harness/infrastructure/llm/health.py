"""
Connectivity health tracker for MyLLM. Requested behaviour:

  "auto check MyLLM connectivity twice; if connected, good — use it.
   if not connected, wait 60s and try once more. if still down (timeout /
   server error), skip MyLLM for this chain call and go straight to the
   next backend."

This is a small in-memory, thread-safe cache of the last known health state,
so we don't re-run this two-attempt probe on every single generate() call —
only when we don't have a recent-enough result. While MyLLM is marked DOWN,
the chain skips straight past it until COOLDOWN_SECONDS has elapsed, at
which point the next call re-probes it once.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from rag_harness.config.settings import MyLLMSettings

logger = logging.getLogger("rag_harness.llm.myllm_health")


class MyLLMHealthMonitor:
    def __init__(self, settings: MyLLMSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._is_up: bool | None = None  # None = never checked yet
        self._last_checked_at: float = 0.0

    def _probe_once(self) -> bool:
        """Single lightweight reachability probe (small request, short timeout)."""
        try:
            resp = requests.post(
                self.settings.API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.AUTH_TOKEN}",
                },
                json={
                    "model": self.settings.MODEL_NAME,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 4,
                },
                timeout=self.settings.HEALTH_CHECK_TIMEOUT_SECONDS,
            )
            return resp.status_code < 500
        except requests.exceptions.RequestException as e:
            logger.debug(f"myllm health probe failed: {e}")
            return False

    def is_available(self) -> bool:
        """Returns True if myllm should be tried for this call. Runs the
        two-attempt check only when we have no cached result or the cooldown
        since the last DOWN result has elapsed."""
        with self._lock:
            now = time.monotonic()
            cooldown_elapsed = (now - self._last_checked_at) >= self.settings.HEALTH_CHECK_COOLDOWN_SECONDS

            if self._is_up is True:
                return True
            if self._is_up is False and not cooldown_elapsed:
                logger.info(
                    f"myllm marked DOWN — skipping (re-checking again in "
                    f"{self.settings.HEALTH_CHECK_COOLDOWN_SECONDS - (now - self._last_checked_at):.0f}s)"
                )
                return False

            # Never checked yet, or cooldown elapsed — run the probe.
            logger.info(f"Checking myllm connectivity ({self.settings.HEALTH_CHECK_ATTEMPTS} attempts)...")
            up = False
            for attempt in range(1, self.settings.HEALTH_CHECK_ATTEMPTS + 1):
                up = self._probe_once()
                if up:
                    logger.info(f"myllm connectivity OK (attempt {attempt}/{self.settings.HEALTH_CHECK_ATTEMPTS})")
                    break
                logger.warning(f"myllm connectivity check failed (attempt {attempt}/{self.settings.HEALTH_CHECK_ATTEMPTS})")

            self._is_up = up
            self._last_checked_at = time.monotonic()
            if not up:
                logger.warning(
                    f"myllm unreachable after {self.settings.HEALTH_CHECK_ATTEMPTS} attempts — "
                    f"falling back to next backend in chain for {self.settings.HEALTH_CHECK_COOLDOWN_SECONDS}s"
                )
            return up

    def report_failure(self) -> None:
        """Called by the chain when a live generate() call to myllm fails
        (e.g. mid-cooldown-window a call still got through the health check
        but then timed out) — marks it down immediately, starting a fresh cooldown."""
        with self._lock:
            self._is_up = False
            self._last_checked_at = time.monotonic()

    def report_success(self) -> None:
        with self._lock:
            self._is_up = True
            self._last_checked_at = time.monotonic()
