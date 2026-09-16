"""Celery tasks for the detection app (Task 9.1, Requirement 6).

Implements the async orchestrator task `analyze_session` coordinating visual and audio
sub-pipelines, multi-modal fusion, and persistence.

The orchestrator branches on the session's ``media_kind``:

* **video**  — full multimodal pipeline (frames + faces -> visual model,
  audio track -> audio model, then fusion of both likelihoods).
* **image**  — visual branch only (the single still image is decoded, faces are
  isolated and scored; no audio analysis, fusion uses the visual likelihood).
* **audio**  — audio branch only (the audio file is loaded directly, the
  log-mel spectrogram is scored, and a spectrogram attention map is streamed as
  the XAI explanation; no visual analysis).

In every case the FusionEngine handles the missing-modality case: fusion of a
single likelihood is that likelihood, so image/audio uploads still produce a
score and a label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

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
    and capped at 3 retries (Requirements 6.3, 6.5, 6.6). The media kind recorded
    on the session decides which branches run.
    """
    try:
        session = AnalysisSession.objects.get(id=session_id)
    except AnalysisSession.DoesNotExist:
        logger.error("AnalysisSession %s does not exist", session_id)
        return "SESSION_NOT_FOUND"

    if session.status == AnalysisSession.Status.CANCELED:
        return "CANCELED"

    session.status = AnalysisSession.Status.PROCESSING
    session.started_at = timezone.now()
    session.save(update_fields=["status", "started_at"])

    # Stream initial 0% progress (Requirement 5.3, 7.1)
    StreamService.publish_event(
        session_id,
        "progress",
        {"progress_percent": 0.0, "status": "PROCESSING"},
    )

    media_dir = session_media_dir(session.id)
    # Sort the candidates so the analyzed file is a deterministic function of
    # the upload (unordered globbing could pick a different file between runs,
    # which made the same video report different scores).
    media_files = sorted(p for p in media_dir.glob("*") if p.is_file()) if media_dir.exists() else []
    if not media_files:
        session.status = AnalysisSession.Status.FAILED
        session.save(update_fields=["status"])
        StreamService.publish_event(
            session_id,
            "error",
            {"error_code": "NO_MEDIA_FOUND", "message": "Media file missing"},
        )
        return "NO_MEDIA_FOUND"

    media_path = str(media_files[0])
    media_kind = session.media_kind or AnalysisSession.MediaKind.VIDEO

    visual_likelihood: float | None = None
    audio_likelihood: float | None = None
    outcome = None  # visual extraction outcome (drives the XAI pass)

    # ------------------------------------------------------------------
    # 1. Visual Pipeline (Requirement 2) — skipped for audio-only uploads
    # ------------------------------------------------------------------
    if media_kind != AnalysisSession.MediaKind.AUDIO:
        try:
            extractor = FrameExtractor()
            if media_kind == AnalysisSession.MediaKind.IMAGE:
                outcome = extractor.extract_visual_signal_from_image(media_path)
            else:
                outcome = extractor.extract_visual_signal(
                    media_path,
                    min_fps=getattr(settings, "FRAME_MIN_FPS", 1.0),
                    max_frames=getattr(settings, "FRAME_SAMPLE_MAX_FRAMES", None),
                )
            # Deterministic face ordering: analysis order never depends on detector
            # return order (Requirement 2 / reproducibility).
            outcome.faces.sort(key=lambda f: (f.frame_index, f.y, f.x))

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

    # ------------------------------------------------------------------
    # 2. Audio Pipeline (Requirement 3) — skipped for image-only uploads
    # ------------------------------------------------------------------
    if media_kind != AnalysisSession.MediaKind.IMAGE:
        try:
            audio_extractor = AudioExtractor()
            a_outcome = audio_extractor.extract_audio_signal(media_path)

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

    # ------------------------------------------------------------------
    # 3. Multi-Modal Fusion (Requirement 4) — single-modality inputs fuse
    #    to that modality's likelihood; FusionEngine handles it.
    # ------------------------------------------------------------------
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
            # The model column is CSV; FusionEngine yields a list.
            "modalities_used": ",".join(fusion_outcome.modalities_used),
            "inconclusive": fusion_outcome.inconclusive,
            "threshold_used": fusion_outcome.threshold_used,
        },
    )

    # 4. Terminal status update & Stream result
    if fusion_outcome.inconclusive:
        session.status = AnalysisSession.Status.INCONCLUSIVE
    else:
        session.status = AnalysisSession.Status.COMPLETED

    session.completed_at = timezone.now()
    session.save(update_fields=["status", "completed_at"])

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
            # Which input kind was analyzed, so the UI can present the right
            # pipeline and evidence view.
            "media_kind": media_kind,
            "source_ref": session.source_ref,
            # Per-modality evidence so the dashboard can explain the fused
            # probability instead of showing a single opaque number.
            "visual_likelihood": visual_likelihood,
            "audio_likelihood": audio_likelihood,
            "visual_state": _visual_state(session),
            "audio_state": _audio_state(session),
            "frames_analyzed": _frames_analyzed(session),
            "faces_isolated": _faces_isolated(session),
            "weights": {"visual": weight_visual, "audio": weight_audio},
            "threshold": fusion_outcome.threshold_used,
        },
    )

    # 4b. XAI artifacts (Requirement 11)
    # Visual inputs: ORIGINAL / HEATMAP / OVERLAY triple for representative
    # frames. Audio inputs: the mel-spectrogram attention map rendered over the
    # spectrogram itself. Heatmap failure never affects the persisted
    # classification (Requirement 11.2) — errors are logged and streamed only.
    try:
        _generate_xai(session_id, media_kind, outcome, audio_likelihood)
    except Exception as exc:  # noqa: BLE001
        logger.warning("XAI pass failed for session %s: %s", session_id, exc)

    # 5. Media Purge (GDPR Requirement 9)
    try:
        from detection.services.purger import MediaPurger
        MediaPurger.purge(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Media purge exception for session %s: %s", session_id, exc)

    return session.status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _visual_state(session: AnalysisSession) -> str | None:
    visual = getattr(session, "visual_result", None)
    return visual.state if visual else None


def _audio_state(session: AnalysisSession) -> str | None:
    audio = getattr(session, "audio_result", None)
    return audio.state if audio else None


def _frames_analyzed(session: AnalysisSession) -> int | None:
    visual = getattr(session, "visual_result", None)
    return visual.frames_analyzed if visual else None


def _faces_isolated(session: AnalysisSession) -> int | None:
    visual = getattr(session, "visual_result", None)
    return visual.faces_isolated if visual else None


def _generate_xai(
    session_id: str,
    media_kind: str,
    outcome: Any,
    audio_likelihood: float | None,
) -> None:
    """Generate the XAI artifacts appropriate for the analyzed media kind."""
    if not getattr(settings, "HEATMAP_ENABLED", False):
        return

    if media_kind in (AnalysisSession.MediaKind.VIDEO, AnalysisSession.MediaKind.IMAGE):
        if outcome is None or not getattr(outcome, "has_visual_signal", False):
            return
        faces = list(getattr(outcome, "faces", []) or [])
        if not faces:
            return

        from detection.services.xai import XAIGenerator  # noqa: PLC0415

        visual_model_for_xai = VisualModel()
        max_frames = getattr(settings, "HEATMAP_MAX_FRAMES", 3)
        for face in faces[:max_frames]:
            XAIGenerator.generate_heatmap(
                session_id=session_id,
                frame_id=(
                    f"frame_{face.frame_index:04d}"
                    if media_kind == AnalysisSession.MediaKind.VIDEO
                    else "image_face_0001"
                ),
                activation_matrix=visual_model_for_xai.activation_map(face),
                frame_image=face.image,
            )
        return

    # AUDIO: render the mel-spectrogram attention map over the spectrogram.
    if audio_likelihood is None:
        return
    try:
        from detection.services.xai import generate_audio_xai  # noqa: PLC0415

        generate_audio_xai(session_id=session_id, likelihood=audio_likelihood)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio XAI generation failed for session %s: %s", session_id, exc)
