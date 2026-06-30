"""Frame_Extractor: frame sampling and face isolation (Task 4.1, Requirement 2).

This module implements the design's ``Frame_Extractor`` interface:

    extract_frames(video_path, min_fps=1.0) -> Iterator[Frame]
    isolate_faces(frame, min_area_ratio=0.05) -> list[FaceRegion]

Design intent (design.md → Components and Interfaces → Frame_Extractor):

* Decode and sample frames at **>= 1 fps** of video duration (Requirement 2.1).
* Isolate only facial regions occupying **>= 5% of the frame area** (Requirement 2.2).
* When **no face** is isolated across all extracted frames, the session is
  ``NO_VISUAL_SIGNAL`` (Requirement 2.4).
* When extraction **fails before producing any frame**, the session is
  ``VISUAL_ERROR`` with an error indication (Requirement 2.5).

Testability decision
---------------------
The two pieces of business logic that the acceptance criteria pin down — the
sampling-rate computation (2.1) and the 5% area thresholding (2.2) — are pure
arithmetic. To keep them unit- and property-testable **without** real OpenCV,
MTCNN/MediaPipe, or video files, the heavy back ends are injected:

* a **decoder** callable ``video_path -> DecodedVideo`` (frame source), and
* a **detector** callable ``Frame -> list[FaceCandidate]`` (face detector).

Tests pass in lightweight fakes (e.g. :class:`InMemoryDecodedVideo` and a plain
function returning :class:`FaceCandidate` boxes). In production the defaults are
real OpenCV / MTCNN back ends, imported **lazily and guarded** so that importing
this module never fails when the ML/CV libraries are absent (they are only
required at the moment a real decode/detect is actually requested).

The ``VisualSignalState`` values mirror ``detection.models.VisualResult.State``
exactly (``OK`` / ``NO_VISUAL_SIGNAL`` / ``VISUAL_ERROR``) so the orchestrator
can persist :class:`ExtractionOutcome` to ``VisualResult`` directly. This module
deliberately does **not** import the Django model, keeping the logic importable
without a configured Django/ORM environment (see ``backend/conftest.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, List, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class VisualSignalState(str, Enum):
    """Visual-analysis signal state for a session.

    Values mirror ``detection.models.VisualResult.State`` so an
    :class:`ExtractionOutcome` maps straight onto the persisted ``VisualResult``.
    """

    OK = "OK"
    NO_VISUAL_SIGNAL = "NO_VISUAL_SIGNAL"
    VISUAL_ERROR = "VISUAL_ERROR"


@dataclass(frozen=True)
class Frame:
    """A single sampled video frame.

    ``image`` carries the decoded pixel data in production (e.g. a NumPy array)
    and is ``None`` in pure-logic tests, which only exercise geometry/sampling.
    """

    index: int  # frame index within the original stream
    timestamp: float  # seconds from start of the video
    width: int
    height: int
    image: Any = None

    @property
    def area(self) -> int:
        """Total pixel area of the frame."""
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class FaceCandidate:
    """A detected face bounding box, before the 5%-area filter is applied.

    This is what a face *detector* yields. Produced either by the real
    MTCNN/MediaPipe back end or by a test fake.
    """

    x: int
    y: int
    width: int
    height: int
    image: Any = None

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class FaceRegion:
    """An isolated facial region that passed the 5%-area threshold (Req 2.2)."""

    frame_index: int
    x: int
    y: int
    width: int
    height: int
    frame_width: int
    frame_height: int
    image: Any = None

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def frame_area(self) -> int:
        return max(0, self.frame_width) * max(0, self.frame_height)

    @property
    def area_ratio(self) -> float:
        """Fraction of the frame occupied by this face region, in [0.0, 1.0]."""
        fa = self.frame_area
        if fa <= 0:
            return 0.0
        return self.area / fa


@dataclass
class ExtractionOutcome:
    """Result of the full visual-extraction pass for a session.

    Maps directly onto ``VisualResult`` fields the orchestrator persists:
    ``state`` -> ``VisualResult.state``, ``frames_analyzed`` ->
    ``frames_analyzed``, ``faces_isolated`` -> ``faces_isolated``,
    ``error_detail`` -> ``error_detail``. ``faces`` carries the isolated
    regions forward to the Visual_Model (Task 6.1).
    """

    state: VisualSignalState
    frames_analyzed: int = 0
    faces_isolated: int = 0
    faces: List[FaceRegion] = field(default_factory=list)
    error_detail: str = ""

    @property
    def has_visual_signal(self) -> bool:
        return self.state is VisualSignalState.OK


# ---------------------------------------------------------------------------
# Back-end abstractions (injectable; real implementations are lazy/guarded)
# ---------------------------------------------------------------------------


@runtime_checkable
class DecodedVideo(Protocol):
    """A decoded video: duration, native frame rate, and a raw-frame source.

    The default OpenCV back end and the in-memory test fake both satisfy this
    protocol. ``raw_frames`` yields frames in capture order; the extractor
    decides which ones to keep based on the target sampling rate.
    """

    @property
    def duration_seconds(self) -> float: ...

    @property
    def frame_rate(self) -> float: ...

    def raw_frames(self) -> Iterable[Frame]: ...


# A decoder turns a path into a DecodedVideo. A detector turns a Frame into
# candidate face boxes. Both are injectable for testing.
Decoder = Callable[[str], DecodedVideo]
Detector = Callable[[Frame], Iterable[FaceCandidate]]


@dataclass
class InMemoryDecodedVideo:
    """A :class:`DecodedVideo` backed by an in-memory list of frames.

    Used by tests (and any non-file caller) to exercise the sampling and
    isolation logic without decoding a real video file.
    """

    duration_seconds: float
    frame_rate: float
    frames: List[Frame] = field(default_factory=list)

    def raw_frames(self) -> Iterable[Frame]:
        return list(self.frames)


# ---------------------------------------------------------------------------
# FrameExtractor
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised by a decoder when a video cannot be decoded at all."""


class FrameExtractor:
    """Sample frames at >= ``min_fps`` and isolate faces >= ``min_area_ratio``.

    Parameters
    ----------
    decoder:
        Callable ``video_path -> DecodedVideo``. Defaults to a lazily-imported
        OpenCV back end (only imported when a real decode is attempted).
    detector:
        Callable ``Frame -> Iterable[FaceCandidate]``. Defaults to a
        lazily-imported MTCNN back end.
    """

    def __init__(
        self,
        decoder: Decoder | None = None,
        detector: Detector | None = None,
    ) -> None:
        self._decoder = decoder if decoder is not None else _default_decoder
        self._detector = detector if detector is not None else _default_detector

    # -- sampling (Requirement 2.1) -----------------------------------------

    @staticmethod
    def _sampling_step(frame_rate: float, min_fps: float) -> int:
        """Number of native frames to skip between samples.

        Choosing ``step = floor(frame_rate / min_fps)`` (>= 1) guarantees an
        effective sampling rate of at least ``min_fps`` frames per second, which
        yields at least ``floor(duration)`` frames for ``min_fps == 1.0``
        (Requirement 2.1 / Property 3).
        """
        if min_fps <= 0:
            raise ValueError("min_fps must be > 0")
        if frame_rate <= 0:
            # Unknown/degenerate frame rate: keep every available frame.
            return 1
        return max(1, int(frame_rate // min_fps))

    def extract_frames(self, video_path: str, min_fps: float = 1.0) -> Iterator[Frame]:
        """Yield frames sampled at **>= ``min_fps``** of the video duration.

        Selects every ``step``-th decoded frame so the effective rate is at
        least ``min_fps`` (Requirement 2.1). Timestamps are derived from the
        native frame rate when available.

        Raises :class:`ExtractionError` if the decoder cannot produce a decoded
        video; the caller (:meth:`extract_visual_signal`) turns a failure with
        zero produced frames into the ``VISUAL_ERROR`` state (Requirement 2.5).
        """
        decoded = self._decoder(video_path)
        frame_rate = float(getattr(decoded, "frame_rate", 0.0) or 0.0)
        step = self._sampling_step(frame_rate, min_fps)

        for position, raw in enumerate(decoded.raw_frames()):
            if position % step != 0:
                continue
            timestamp = (position / frame_rate) if frame_rate > 0 else float(position)
            yield Frame(
                index=position,
                timestamp=timestamp,
                width=raw.width,
                height=raw.height,
                image=raw.image,
            )

    # -- face isolation (Requirement 2.2) -----------------------------------

    def isolate_faces(
        self, frame: Frame, min_area_ratio: float = 0.05
    ) -> list[FaceRegion]:
        """Isolate exactly the faces occupying **>= ``min_area_ratio``** of the frame.

        Runs the (injected) detector and keeps only candidate boxes whose area
        is at least ``min_area_ratio`` of the frame area (Requirement 2.2 /
        Property 4). Candidates below the threshold are discarded; no others are
        kept.
        """
        frame_area = frame.area
        if frame_area <= 0:
            return []

        regions: list[FaceRegion] = []
        for candidate in self._detector(frame):
            ratio = candidate.area / frame_area
            if ratio >= min_area_ratio:
                regions.append(
                    FaceRegion(
                        frame_index=frame.index,
                        x=candidate.x,
                        y=candidate.y,
                        width=candidate.width,
                        height=candidate.height,
                        frame_width=frame.width,
                        frame_height=frame.height,
                        image=candidate.image,
                    )
                )
        return regions

    # -- orchestrator-facing pass (Requirements 2.4, 2.5) -------------------

    def extract_visual_signal(
        self,
        video_path: str,
        min_fps: float = 1.0,
        min_area_ratio: float = 0.05,
    ) -> ExtractionOutcome:
        """Run the full extract-and-isolate pass and classify the visual signal.

        Returns an :class:`ExtractionOutcome` the orchestrator persists to
        ``VisualResult``:

        * ``VISUAL_ERROR`` — extraction failed before producing any frame
          (Requirement 2.5). ``error_detail`` identifies the failure.
        * ``NO_VISUAL_SIGNAL`` — frames were produced but no face met the 5%
          threshold across all of them (Requirement 2.4).
        * ``OK`` — at least one facial region was isolated.

        In every case audio analysis can continue independently; this method
        never raises for a decode/detect failure, it encodes it as state.
        """
        frames_analyzed = 0
        faces: list[FaceRegion] = []

        try:
            for frame in self.extract_frames(video_path, min_fps=min_fps):
                frames_analyzed += 1
                faces.extend(self.isolate_faces(frame, min_area_ratio=min_area_ratio))
        except Exception as exc:  # noqa: BLE001 - failures are encoded as state
            if frames_analyzed == 0:
                # Extraction failed before any frame was produced (Req 2.5).
                return ExtractionOutcome(
                    state=VisualSignalState.VISUAL_ERROR,
                    frames_analyzed=0,
                    faces_isolated=0,
                    faces=[],
                    error_detail=f"frame extraction failed: {exc}",
                )
            # Frames were already produced before the failure: proceed with what
            # we have rather than discarding a partial visual signal.

        if not faces:
            # Frames produced but no qualifying face anywhere (Req 2.4).
            return ExtractionOutcome(
                state=VisualSignalState.NO_VISUAL_SIGNAL,
                frames_analyzed=frames_analyzed,
                faces_isolated=0,
                faces=[],
                error_detail="",
            )

        return ExtractionOutcome(
            state=VisualSignalState.OK,
            frames_analyzed=frames_analyzed,
            faces_isolated=len(faces),
            faces=faces,
            error_detail="",
        )


# ---------------------------------------------------------------------------
# Default real back ends (lazy + guarded so imports never fail without the libs)
# ---------------------------------------------------------------------------


def _default_decoder(video_path: str) -> DecodedVideo:
    """Decode ``video_path`` with OpenCV (imported lazily and guarded).

    OpenCV is only required at the moment a real decode is attempted, so this
    module imports cleanly in environments where ``opencv-python`` is absent
    (e.g. the pure-logic test harness).
    """
    try:
        import cv2  # type: ignore  # noqa: PLC0415 - intentional lazy import
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(
            "OpenCV (opencv-python) is required to decode video files but is "
            "not available. Inject a custom decoder for testing."
        ) from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ExtractionError(f"could not open video: {video_path}")

    frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / frame_rate) if frame_rate > 0 else 0.0

    def _iter_frames() -> Iterator[Frame]:
        position = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                yield Frame(
                    index=position,
                    timestamp=(position / frame_rate) if frame_rate > 0 else float(position),
                    width=int(width),
                    height=int(height),
                    image=image,
                )
                position += 1
        finally:
            capture.release()

    return InMemoryDecodedVideo(
        duration_seconds=duration,
        frame_rate=frame_rate,
        frames=list(_iter_frames()),
    )


def _default_detector(frame: Frame) -> list[FaceCandidate]:
    """Detect faces in ``frame`` with MTCNN (imported lazily and guarded)."""
    try:
        from mtcnn import MTCNN  # type: ignore  # noqa: PLC0415 - lazy import
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(
            "MTCNN is required to detect faces but is not available. Inject a "
            "custom detector for testing."
        ) from exc

    if frame.image is None:
        return []

    detector = MTCNN()
    candidates: list[FaceCandidate] = []
    for detection in detector.detect_faces(frame.image):
        box = detection.get("box")
        if not box:
            continue
        x, y, width, height = box
        candidates.append(
            FaceCandidate(x=int(x), y=int(y), width=int(width), height=int(height))
        )
    return candidates
