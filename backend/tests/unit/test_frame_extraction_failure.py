"""Unit tests for visual extraction-failure handling (Task 4.4, Requirement 2.5).

Example-based coverage of :meth:`FrameExtractor.extract_visual_signal` when the
visual pipeline cannot produce a usable signal. These tests inject lightweight
fakes for the decoder/detector back ends, so they run purely on logic without
OpenCV, MTCNN, a GPU, Django, or any real video file (see ``backend/conftest.py``).

Two distinct, non-overlapping states are pinned down:

* ``VISUAL_ERROR`` (Requirement 2.5) — extraction fails *before any frame is
  produced*. The injected decoder raises on first use. The method must encode
  the failure as state (never raise), report ``frames_analyzed == 0`` and
  ``faces_isolated == 0``, carry a non-empty ``error_detail`` that identifies the
  failure, and expose ``has_visual_signal is False`` so the orchestrator
  continues with audio analysis independently.
* ``NO_VISUAL_SIGNAL`` (Requirement 2.4) — frames *are* produced but no face
  qualifies. This is asserted here only to confirm the two states are distinct:
  a produced-but-empty pass is **not** an error.

Requirement 2.5 also says audio analysis continues after a visual failure.
Because :meth:`extract_visual_signal` swallows the decode/detect exception and
returns an :class:`ExtractionOutcome` (rather than propagating), the orchestrator
is free to run audio independently. We assert that the call returns normally and
that ``has_visual_signal`` is ``False`` — the contract the orchestrator relies on
to proceed with audio.
"""

from __future__ import annotations

import pytest

from detection.processing.frame_extractor import (
    ExtractionError,
    FaceCandidate,
    Frame,
    FrameExtractor,
    InMemoryDecodedVideo,
    VisualSignalState,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Injectable fakes (no OpenCV/MTCNN, no real video).
# ---------------------------------------------------------------------------
def _raising_decoder(exc: Exception):
    """Return a decoder callable that raises ``exc`` before yielding any frame."""

    def decoder(_video_path: str):  # noqa: ANN202 - test fake
        raise exc

    return decoder


def _video_with_frames(count: int, *, width: int = 100, height: int = 100):
    """Return a decoder yielding ``count`` plain frames at 1 fps (no faces injected)."""
    frames = [
        Frame(index=i, timestamp=float(i), width=width, height=height)
        for i in range(count)
    ]
    decoded = InMemoryDecodedVideo(
        duration_seconds=float(count), frame_rate=1.0, frames=frames
    )

    def decoder(_video_path: str):  # noqa: ANN202 - test fake
        return decoded

    return decoder


def _no_qualifying_detector(_frame: Frame) -> list[FaceCandidate]:
    """Detector that returns only sub-threshold candidates (below 5% of frame area).

    On a 100x100 (10000 px) frame, a 10x10 (100 px) box is 1% of the area, well
    under the 5% threshold, so :meth:`isolate_faces` discards it → NO_VISUAL_SIGNAL.
    """
    return [FaceCandidate(x=0, y=0, width=10, height=10)]


# ---------------------------------------------------------------------------
# Requirement 2.5: extraction failure before any frame → VISUAL_ERROR.
# ---------------------------------------------------------------------------
class TestExtractionFailureVisualError:
    """A decoder that fails before producing a frame yields VISUAL_ERROR."""

    def test_extraction_error_before_any_frame_is_visual_error(self) -> None:
        """ExtractionError on decode → VISUAL_ERROR with a non-empty error detail."""
        extractor = FrameExtractor(
            decoder=_raising_decoder(ExtractionError("could not open video: clip.mp4"))
        )

        # The method must encode failure as state, never propagate the exception.
        outcome = extractor.extract_visual_signal("clip.mp4")

        assert outcome.state is VisualSignalState.VISUAL_ERROR
        assert outcome.frames_analyzed == 0
        assert outcome.faces_isolated == 0
        assert outcome.faces == []
        # Error indication is present and identifies the failure (Req 2.5).
        assert isinstance(outcome.error_detail, str)
        assert outcome.error_detail != ""
        assert "could not open video: clip.mp4" in outcome.error_detail
        # Orchestrator contract: no visual signal, so audio proceeds independently.
        assert outcome.has_visual_signal is False

    def test_generic_exception_before_any_frame_is_visual_error(self) -> None:
        """Any decoder exception (not just ExtractionError) is encoded as VISUAL_ERROR."""
        extractor = FrameExtractor(
            decoder=_raising_decoder(RuntimeError("decoder backend crashed"))
        )

        outcome = extractor.extract_visual_signal("clip.mp4")

        assert outcome.state is VisualSignalState.VISUAL_ERROR
        assert outcome.frames_analyzed == 0
        assert outcome.faces_isolated == 0
        assert outcome.error_detail != ""
        assert "decoder backend crashed" in outcome.error_detail
        assert outcome.has_visual_signal is False

    def test_visual_failure_does_not_block_audio(self) -> None:
        """Req 2.5: a visual failure returns normally so audio analysis can continue.

        We model the orchestrator's behaviour: it calls extract_visual_signal,
        sees ``has_visual_signal is False`` (without an exception escaping), and
        then runs audio analysis independently. A raised exception would abort
        that flow, so the contract is "return, don't raise".
        """
        extractor = FrameExtractor(
            decoder=_raising_decoder(ExtractionError("boom"))
        )

        audio_ran = False

        # No exception should escape extract_visual_signal.
        outcome = extractor.extract_visual_signal("clip.mp4")
        if not outcome.has_visual_signal:
            # Orchestrator proceeds with audio regardless of visual outcome.
            audio_ran = True

        assert outcome.state is VisualSignalState.VISUAL_ERROR
        assert audio_ran is True


# ---------------------------------------------------------------------------
# Requirement 2.4: frames produced but no qualifying face → NO_VISUAL_SIGNAL.
# Asserted here to confirm NO_VISUAL_SIGNAL and VISUAL_ERROR are distinct.
# ---------------------------------------------------------------------------
class TestNoVisualSignalIsDistinctFromError:
    """A produced-but-empty pass is NO_VISUAL_SIGNAL, not VISUAL_ERROR."""

    def test_frames_with_no_qualifying_face_is_no_visual_signal(self) -> None:
        """Frames decoded, detector returns sub-threshold boxes → NO_VISUAL_SIGNAL."""
        extractor = FrameExtractor(
            decoder=_video_with_frames(3),
            detector=_no_qualifying_detector,
        )

        outcome = extractor.extract_visual_signal("clip.mp4", min_area_ratio=0.05)

        assert outcome.state is VisualSignalState.NO_VISUAL_SIGNAL
        # Frames were produced (distinguishing this from the VISUAL_ERROR path).
        assert outcome.frames_analyzed == 3
        assert outcome.faces_isolated == 0
        assert outcome.faces == []
        # NO_VISUAL_SIGNAL is not an error: no error_detail is recorded.
        assert outcome.error_detail == ""
        assert outcome.has_visual_signal is False

    def test_no_visual_signal_and_visual_error_are_different_states(self) -> None:
        """The two failure-ish states are genuinely distinct enum values."""
        error_outcome = FrameExtractor(
            decoder=_raising_decoder(ExtractionError("boom"))
        ).extract_visual_signal("clip.mp4")

        empty_outcome = FrameExtractor(
            decoder=_video_with_frames(2),
            detector=_no_qualifying_detector,
        ).extract_visual_signal("clip.mp4")

        assert error_outcome.state is VisualSignalState.VISUAL_ERROR
        assert empty_outcome.state is VisualSignalState.NO_VISUAL_SIGNAL
        assert error_outcome.state is not empty_outcome.state
