"""Typed environment-variable helpers for settings.

These helpers keep ``settings.py`` declarative: every setting reads from the
environment with a sensible default and is range-clamped where the design
defines bounds (e.g. ``CELERY_WORKER_CONCURRENCY`` in 1-64, thresholds in
[0.0, 1.0]).

``load_dotenv`` is a thin wrapper that uses ``python-dotenv`` when installed but
degrades gracefully (no-op) when it is not, so the project can be imported and
``manage.py check`` can run without the optional dependency present.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Load environment variables from ``path`` if python-dotenv is available.

    Falls back to a no-op when the package is missing so that core Django
    commands do not hard-depend on it.
    """
    try:
        from dotenv import load_dotenv as _load_dotenv
    except Exception:  # pragma: no cover - optional dependency
        return
    if path.exists():
        _load_dotenv(path)


def env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value is not None and value != "" else default


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(key: str, default: str, sep: str = ",") -> list[str]:
    raw = env_str(key, default)
    return [item.strip() for item in raw.split(sep) if item.strip()]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def env_int_clamped(key: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(key)
    try:
        value = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        value = default
    return int(_clamp(value, low, high))


def env_float_clamped(key: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(key)
    try:
        value = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        value = default
    return _clamp(value, low, high)
