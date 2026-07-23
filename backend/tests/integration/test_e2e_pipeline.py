"""End-to-end happy path integration test (Task 18.3, Requirements 1, 2, 3, 4, 6, 9).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from detection.models import AnalysisSession, FusionResult, PurgeRecord
from detection.services.upload import UploadService
from detection.storage import session_media_dir
from detection.tasks import analyze_session

pytestmark = [pytest.mark.django_db, pytest.mark.integration]


def test_end_to_end_analysis_and_purge_flow() -> None:
    """Full E2E test: upload video -> session created -> analyze_session task runs -> score produced -> media purged."""
    upload = SimpleUploadedFile("sample.mp4", b"valid_video_bytes", content_type="video/mp4")

    # 1. Upload
    service = UploadService(
        probe=lambda _p: True,
        enqueuer=lambda _s: True,  # Don't invoke real Celery worker in test
    )
    result = service.submit_file(upload)
    assert result["accepted"] is True
    session_id = result["session_id"]
    assert session_id is not None

    # Write a dummy video file into media_dir so analyze_session finds it
    media_dir = session_media_dir(session_id)
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir / "sample.mp4").write_bytes(b"dummy_bytes")

    # 2. Run analysis orchestrator task
    # Mock frame and audio extractors to yield OK signal without OpenCV/FFmpeg
    from detection.processing import AudioExtractionOutcome, AudioSignalState, ExtractionOutcome, VisualSignalState
    from detection.processing.frame_extractor import FaceRegion

    dummy_face = FaceRegion(0, 0, 0, 50, 50, 100, 100)
    mock_v = ExtractionOutcome(state=VisualSignalState.OK, frames_analyzed=10, faces_isolated=1, faces=[dummy_face])
    mock_a = AudioExtractionOutcome(state=AudioSignalState.OK, has_audio_track=True, spectrograms=[None])

    with patch("detection.tasks.FrameExtractor.extract_visual_signal", return_value=mock_v):
        with patch("detection.tasks.AudioExtractor.extract_audio_signal", return_value=mock_a):
            with patch("detection.tasks.VisualModel.infer_face", return_value=0.85):
                with patch("detection.tasks.AudioModel.infer", return_value=0.75):
                    status_out = analyze_session(session_id)

    # 3. Verify classification results
    assert status_out == "COMPLETED"
    session = AnalysisSession.objects.get(id=session_id)
    assert session.status == AnalysisSession.Status.COMPLETED

    fusion = FusionResult.objects.get(session=session)
    assert fusion.score == pytest.approx(0.81)  # 0.85*0.6 + 0.75*0.4 = 0.81
    assert fusion.label == FusionResult.Label.DEEPFAKE

    # 4. Verify Media Purge post-analysis (GDPR Requirement 9)
    assert not media_dir.exists()
    assert session.media_state == AnalysisSession.MediaState.PURGED
    purge_rec = PurgeRecord.objects.get(session=session)
    assert purge_rec.status == PurgeRecord.Status.SUCCESS
