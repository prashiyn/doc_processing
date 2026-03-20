"""
Groq API rate limiting: RPM (requests per minute) and RPD (requests per day).
Uses config_dir/groq_limits.yaml; blocks until a request is allowed when running in a loop.
See https://console.groq.com/docs/rate-limits and https://console.groq.com/settings/limits
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import yaml

from doc_processing.config import get_config_dir

GROQ_PREFIX = "groq/"


def _load_groq_limits() -> tuple[int, int, dict[str, dict[str, int]]]:
    """Return (default_rpm, default_rpd, models_dict)."""
    path = get_config_dir() / "groq_limits.yaml"
    if not path.exists():
        return 30, 1000, {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    default_rpm = int(cfg.get("default_rpm", 30))
    default_rpd = int(cfg.get("default_rpd", 1000))
    models = cfg.get("models") or {}
    return default_rpm, default_rpd, models


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_today_start() -> datetime:
    n = _utc_now()
    return n.replace(hour=0, minute=0, second=0, microsecond=0)


class GroqRateLimiter:
    """
    Tracks requests per minute and per day for Groq models. Call wait_if_needed(model)
    before each request and record_request(model) after; wait_if_needed blocks until
    the request is allowed. Thread-safe.
    """

    def __init__(self):
        self._lock = Lock()
        self._default_rpm, self._default_rpd, self._models = _load_groq_limits()
        self._per_model: dict[str, tuple[deque[float], list[float]]] = {}

    def _get_limits(self, model: str) -> tuple[int, int]:
        entry = self._models.get(model)
        if entry:
            return int(entry.get("rpm", self._default_rpm)), int(entry.get("rpd", self._default_rpd))
        return self._default_rpm, self._default_rpd

    def _get_or_create_tracking(self, model: str) -> tuple[deque[float], list[float]]:
        with self._lock:
            if model not in self._per_model:
                self._per_model[model] = (deque(maxlen=2000), [])
            return self._per_model[model]

    def _prune_minute(self, q: deque[float], rpm: int) -> None:
        cutoff = time.time() - 60.0
        while q and q[0] < cutoff:
            q.popleft()
        while len(q) >= rpm:
            q.popleft()

    def _prune_day(self, day_ts: list[float], rpd: int, day_start_ts: float) -> None:
        while day_ts and day_ts[0] < day_start_ts:
            day_ts.pop(0)
        while len(day_ts) >= rpd:
            day_ts.pop(0)

    def wait_if_needed(self, model: str) -> None:
        """
        Block until one more request is allowed for this model (RPM and RPD).
        Call this immediately before issuing the Groq API request.
        """
        if not model.startswith(GROQ_PREFIX):
            return
        rpm, rpd = self._get_limits(model)
        q, day_ts = self._get_or_create_tracking(model)
        day_start = _utc_today_start().timestamp()

        while True:
            with self._lock:
                now = time.time()
                self._prune_minute(q, rpm)
                self._prune_day(day_ts, rpd, day_start)
                if len(q) < rpm and len(day_ts) < rpd:
                    break
                # At limit: wait for oldest to expire
                wait_sec = 60.0
                if len(q) >= rpm and q:
                    wait_sec = min(wait_sec, max(0, 60.0 - (now - q[0])))
                if len(day_ts) >= rpd and day_ts:
                    next_day = day_start + 86400
                    wait_sec = min(wait_sec, max(0, next_day - now))
            if wait_sec > 0:
                time.sleep(min(wait_sec, 1.0))

    def record_request(self, model: str) -> None:
        """Call after a request has been sent. Thread-safe."""
        if not model.startswith(GROQ_PREFIX):
            return
        now = time.time()
        day_start = _utc_today_start().timestamp()
        q, day_ts = self._get_or_create_tracking(model)
        with self._lock:
            q.append(now)
            while day_ts and day_ts[0] < day_start:
                day_ts.pop(0)
            day_ts.append(now)


_groq_limiter: GroqRateLimiter | None = None


def get_groq_rate_limiter() -> GroqRateLimiter:
    """Singleton Groq rate limiter for use by LLMClient."""
    global _groq_limiter
    if _groq_limiter is None:
        _groq_limiter = GroqRateLimiter()
    return _groq_limiter
