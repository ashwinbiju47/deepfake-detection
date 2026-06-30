"""Transient media store helpers (GDPR-transient handling, Requirement 9).

Uploaded videos and intermediate artifacts are stored under a per-session
subdirectory of ``MEDIA_ROOT`` keyed by ``session_id``. Namespacing by session
id keeps concurrent sessions isolated (design: Concurrency Model) and gives the
Media_Purger a single directory to remove per session.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings


def session_media_dir(session_id: str) -> Path:
    """Return the transient media directory for a given ``session_id``."""
    return Path(settings.MEDIA_ROOT) / str(session_id)
