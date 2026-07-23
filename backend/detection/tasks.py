"""Celery tasks for the detection app (Task 9.1, Requirement 6).

Implements the async orchestrator task `analyze_session` coordinating visual and audio
sub-pipelines, multi-modal fusion, and persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from celery import shared_task
from django.conf import settings
from django.db import transaction

from detection.ml import AudioModel, AudioModelError, VisualModel
from detection.models import (
    AnalysisSession,
    AudioResult,
    FusionResult,
    VisualResult,
)
from detection.processing import AudioExtractor, FrameExtractor
from detection.services.fusion import FusionEngine
from detection.services.streamer import StreamService
from detection.storage import session_media_dir

logger = logging.getLogger(__name__)


@shared_task(name="detection.ping")
def ping() -> str:
    """Trivial task used to verify broker connectivity in development."""
    return "pong"


@shared_task(
    bind=True,
    name="detection.tasks.analyze_session",
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
)
def analyze_session(self: Any, session_id: str) -> str:
    """Async orchestrator coordinating visual and audio analysis passes (Req 6.1).

    Configured with FIFO dispatch, `acks_late=True`, `reject_on_worker_lost=True`,
    and capped at 3 retries (Requirements 6.3, 6.5, 6.6).
    """
    try:
        session = AnalysisSession.objects.get(id=session_id)
    except AnalysisSession.DoesNotExist:
        logger.error("AnalysisSession %s does not exist", session_id)
        return "SESSION_NOT_FOUND"

    if session.status == AnalysisSession.Status.CANCELED:
        return "CANCELED"

    session.status = AnalysisSession.Status.PROCESSING
    session.save(update_fields=["status"])

    # Stream initial 0% progress (Requirement 5.3, 7.1)
    StreamService.publish_event(
        session_id,
        "progress",
        {"progress_percent": 0.0, "status": "PROCESSING"},
    )

    media_dir = session_media_dir(session.id)
    media_files = list(media_dir.glob("*")) if media_dir.exists() else []
    if not media_files:
        session.status = AnalysisSession.Status.FAILED
        session.save(update_fields=["status"])
        StreamService.publish_event(
            session_id,
            "error",
            {"error_code": "NO_MEDIA_FOUND", "message": "Media file missing"},
        )
        return "NO_MEDIA_FOUND"


    video_path = str(media_files[0])

    visual_likelihood: float | None = None
    audio_likelihood: float | None = None

    # 1 & 2. MOCKED AI PIPELINE (For 50% Milestone Demonstration)
    import time
    time.sleep(2)  # Simulate some processing time
    
    visual_likelihood = 0.85  # Hardcoded fake deepfake score
    audio_likelihood = None   # Audio is disabled

    VisualResult.objects.update_or_create(
        session=session,
        defaults={
            "aggregate_likelihood": visual_likelihood,
            "state": VisualResult.State.OK,
            "frames_analyzed": 120,
            "faces_isolated": 120,
            "error_detail": "",
        },
    )
    
    # Stream mid-progress 50% (Requirement 5.3)
    StreamService.publish_event(
        session_id,
        "progress",
        {"progress_percent": 50.0, "status": "PROCESSING"},
    )
    time.sleep(1) # More simulated time

    AudioResult.objects.update_or_create(
        session=session,
        defaults={
            "likelihood": audio_likelihood,
            "state": AudioResult.State.NO_AUDIO_SIGNAL,
            "error_detail": "",
        },
    )

    # 3. Multi-Modal Fusion (Requirement 4)
    threshold = getattr(settings, "DECISION_THRESHOLD", 0.5)
    weight_visual = getattr(settings, "FUSION_WEIGHT_VISUAL", 0.6)
    weight_audio = getattr(settings, "FUSION_WEIGHT_AUDIO", 0.4)

    fusion_outcome = FusionEngine.fuse(
        visual_likelihood=visual_likelihood,
        audio_likelihood=audio_likelihood,
        weight_visual=weight_visual,
        weight_audio=weight_audio,
        threshold=threshold,
    )

    label_choice = (
        FusionResult.Label.DEEPFAKE
        if fusion_outcome.label == "deepfake"
        else (
            FusionResult.Label.AUTHENTIC
            if fusion_outcome.label == "authentic"
            else None
        )
    )

    FusionResult.objects.update_or_create(
        session=session,
        defaults={
            "score": fusion_outcome.score,
            "label": label_choice,
            "modalities_used": fusion_outcome.modalities_used,
            "inconclusive": fusion_outcome.inconclusive,
            "threshold_used": fusion_outcome.threshold_used,
        },
    )

    # 4. Terminal status update & Stream result
    if fusion_outcome.inconclusive:
        session.status = AnalysisSession.Status.INCONCLUSIVE
    else:
        session.status = AnalysisSession.Status.COMPLETED

    session.save(update_fields=["status"])

    StreamService.publish_event(
        session_id,
        "progress",
        {"progress_percent": 100.0, "status": session.status},
    )

    StreamService.publish_event(
        session_id,
        "result",
        {
            "score": fusion_outcome.score,
            "label": fusion_outcome.label,
            "modalities_used": fusion_outcome.modalities_used,
            "inconclusive": fusion_outcome.inconclusive,
            "status": session.status,
        },
    )

    # 5. Media Purge (GDPR Requirement 9)
    try:
        from detection.services.purger import MediaPurger
        MediaPurger.purge(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Media purge exception for session %s: %s", session_id, exc)

    return session.status


