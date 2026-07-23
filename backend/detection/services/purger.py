"""Media_Purger: transient media deletion, verification, and retry (Task 11.1, Requirement 9).

This module implements the design's ``Media_Purger`` interface:

    purge(session_id) -> PurgeRecord

Design intent (design.md -> Components and Interfaces -> Media_Purger):
* Delete uploaded video and intermediate artifacts post-analysis (Requirement 9.1, 9.2, GDPR compliance).
* Verify storage directory is empty (Requirement 9.4).
* Record PurgeRecord in database with retry attempt count and outcome (SUCCESS / FAILED).
* On persistent failure after 3 retries, set status FAILED and raise operator alert (Requirement 9.5).
* Retain non-media metadata (score, label, timestamps) while removing raw video bytes (Requirement 9.3).
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional
from django.db import transaction
from django.utils import timezone

from detection.models import AnalysisSession, PurgeRecord
from detection.storage import session_media_dir

logger = logging.getLogger(__name__)


class MediaPurgeError(Exception):
    """Raised when media purge fails after maximum retries."""


class MediaPurger:
    """Purges transient media files and records audit outcomes."""

    @staticmethod
    def purge(session_id: str, max_retries: int = 3) -> PurgeRecord:
        """Purge media for session_id, verifying directory deletion and recording PurgeRecord."""
        try:
            session = AnalysisSession.objects.get(id=session_id)
        except AnalysisSession.DoesNotExist:
            logger.error("MediaPurger: AnalysisSession %s not found", session_id)
            raise MediaPurgeError(f"Session {session_id} not found")

        media_dir = session_media_dir(session.id)
        attempts = 0
        success = False

        while attempts < max_retries:
            attempts += 1
            try:
                if media_dir.exists():
                    shutil.rmtree(media_dir)
                # Verify empty
                verified_empty = not media_dir.exists()
                if verified_empty:
                    success = True
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MediaPurger attempt %d failed for session %s: %s",
                    attempts,
                    session_id,
                    exc,
                )

        status_choice = (
            PurgeRecord.Status.SUCCESS if success else PurgeRecord.Status.FAILED
        )

        with transaction.atomic():
            record, _created = PurgeRecord.objects.update_or_create(
                session=session,
                defaults={
                    "status": status_choice,
                    "retries": attempts,
                    "verified_empty": success,
                    "purged_at": timezone.now(),
                },
            )
            if success:
                session.media_state = AnalysisSession.MediaState.PURGED
                session.save(update_fields=["media_state"])

        if not success:
            logger.critical(
                "OPERATOR ALERT: Media purge persistently failed for session %s after %d retries!",
                session_id,
                attempts,
            )

        return record
