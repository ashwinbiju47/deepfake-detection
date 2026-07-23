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

    # 1. Visual Pipeline (Requirement 2)
    try:
        extractor = FrameExtractor()
        outcome = extractor.extract_visual_signal(video_path)

        if outcome.has_visual_signal and outcome.faces:
            visual_model = VisualModel()
            face_scores = [
                visual_model.infer_face(face) for face in outcome.faces
            ]
            visual_likelihood = visual_model.aggregate(face_scores)
            v_state = VisualResult.State.OK
        else:
            v_state = (
                VisualResult.State.NO_VISUAL_SIGNAL
                if outcome.state.value == "NO_VISUAL_SIGNAL"
                else VisualResult.State.VISUAL_ERROR
            )
            visual_likelihood = None

        VisualResult.objects.update_or_create(
            session=session,
            defaults={
                "aggregate_likelihood": visual_likelihood,
                "state": v_state,
                "frames_analyzed": outcome.frames_analyzed,
                "faces_isolated": outcome.faces_isolated,
                "error_detail": outcome.error_detail,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Visual pipeline failed for session %s: %s", session_id, exc)
        VisualResult.objects.update_or_create(
            session=session,
            defaults={
                "aggregate_likelihood": None,
                "state": VisualResult.State.VISUAL_ERROR,
                "frames_analyzed": 0,
                "faces_isolated": 0,
                "error_detail": f"visual pipeline exception: {exc}",
            },
        )
        visual_likelihood = None

    # Stream mid-progress 50% (Requirement 5.3)
    StreamService.publish_event(
        session_id,
        "progress",
        {"progress_percent": 50.0, "status": "PROCESSING"},
    )

    # 2. Audio Pipeline (Requirement 3)
    try:
        audio_extractor = AudioExtractor()
        a_outcome = audio_extractor.extract_audio_signal(video_path)

        if a_outcome.has_audio_signal and a_outcome.spectrograms:
            audio_model = AudioModel()
            audio_likelihood = audio_model.infer(a_outcome.spectrograms)
            a_state = AudioResult.State.OK
            a_detail = ""
        else:
            a_state = (
                AudioResult.State.NO_AUDIO_SIGNAL
                if a_outcome.state.value == "NO_AUDIO_SIGNAL"
                else AudioResult.State.AUDIO_ERROR
            )
            audio_likelihood = None
            a_detail = a_outcome.error_detail

        AudioResult.objects.update_or_create(
            session=session,
            defaults={
                "likelihood": audio_likelihood,
                "state": a_state,
                "error_detail": a_detail,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio pipeline failed for session %s: %s", session_id, exc)
        AudioResult.objects.update_or_create(
            session=session,
            defaults={
                "likelihood": None,
                "state": AudioResult.State.AUDIO_ERROR,
                "error_detail": f"audio pipeline exception: {exc}",
            },
        )
        audio_likelihood = None

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


