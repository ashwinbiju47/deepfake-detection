"""Face detection, deterministic inference and frame-sampling regressions.

These tests pin down three defects that made the same upload behave
differently between runs:

1. face detection required MTCNN/TensorFlow, so with neither installed every
   upload degraded to ``NO_VISUAL_SIGNAL`` (no faces, no XAI triple);
2. the default inference back ends returned a constant/nondeterministic value
   instead of a pure function of the input;
3. the OpenCV decoder materialized every decoded frame in memory, so a long
   video exhausts the worker (and the session has to be re-run).
"""

from __future__ import annotations

import numpy as np
import pytest

from detection.processing.frame_extractor import (
    ExtractionError,
    FaceCandidate,
    FaceRegion,
    Frame,
    FrameExtractor,
    InMemoryDecodedVideo,
    _crop_face,
    _dedupe_boxes,
    _get_face_detector,
    _HaarFaceDetector,
    _MTCNNFaceDetector,
    FACE_CROP_SIZE,
)
from detection.ml.audio_model import AudioModel, high_frequency_energy_ratio
from detection.ml.visual_model import (
    VisualModel,
    content_likelihood,
    face_high_frequency_ratio,
)
from detection.processing.audio_extractor import SpectrogramRepresentation

pytestmark = pytest.mark.unit


def _face(image: object = None, x: int = 10, y: int = 20) -> FaceRegion:
    return FaceRegion(
        frame_index=0,
        x=x,
        y=y,
        width=80,
        height=80,
        frame_width=640,
        frame_height=480,
        image=image,
    )


def _gradient_face(seed: int = 0) -> np.ndarray:
    """A deterministic synthetic 224x224 BGR face crop."""
    yy, xx = np.mgrid[0:224, 0:224]
    base = ((xx * 3 + yy * 5 + seed * 37) % 200) + 20
    return np.stack([base, base // 2, 255 - base], axis=-1).astype("uint8")


class TestFaceDetectorFallback:
    def test_haar_fallback_is_used_when_mtcnn_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(self) -> None:  # noqa: ANN001
            raise ExtractionError("TensorFlow is not installed")

        monkeypatch.setattr(_MTCNNFaceDetector, "__init__", _boom)
        import detection.processing.frame_extractor as fe

        monkeypatch.setattr(fe, "_detector_cache", {})
        detector = _get_face_detector()
        assert detector.name == "opencv-haar"

    def test_haar_detector_runs_and_finds_no_faces_in_blank_frame(self) -> None:
        detector = _HaarFaceDetector()
        blank = np.zeros((240, 320, 3), dtype="uint8")
        frame = Frame(index=0, timestamp=0.0, width=320, height=240, image=blank)
        assert detector.detect(frame) == []

    def test_default_detector_returns_candidates(self) -> None:
        blank = np.zeros((240, 320, 3), dtype="uint8")
        frame = Frame(index=0, timestamp=0.0, width=320, height=240, image=blank)
        from detection.processing.frame_extractor import _default_detector

        assert _default_detector(frame) == []

    def test_detected_faces_carry_a_normalized_crop(self) -> None:
        """The XAI ORIGINAL panel needs real pixels, not a placeholder."""
        bgr = (_gradient_face() * 0.2).astype("uint8")
        crop = _crop_face(bgr, 10, 10, 60, 60)
        assert crop is not None
        assert crop.shape == (FACE_CROP_SIZE, FACE_CROP_SIZE, 3)


class TestBoxDeduplication:
    def test_overlapping_boxes_are_merged_deterministically(self) -> None:
        boxes = [(10, 10, 100, 100), (12, 12, 98, 98), (300, 200, 80, 80)]
        merged = _dedupe_boxes(boxes)
        assert len(merged) == 2
        assert merged == _dedupe_boxes(boxes)

    def test_non_overlapping_boxes_are_kept_in_reading_order(self) -> None:
        boxes = [(300, 200, 60, 60), (10, 10, 60, 60)]
        assert _dedupe_boxes(boxes) == [(10, 10, 60, 60), (300, 200, 60, 60)]


class TestDeterministicInference:
    def test_visual_score_is_stable_for_the_same_face(self) -> None:
        """Same video -> same score (regression: differing percentages)."""
        model = VisualModel()
        face = _face(image=_gradient_face())
        scores = [model.infer_face(face) for _ in range(5)]
        assert len(set(scores)) == 1

    def test_visual_score_is_a_fresh_instance_identical(self) -> None:
        face = _face(image=_gradient_face())
        assert VisualModel().infer_face(face) == VisualModel().infer_face(face)

    def test_visual_score_varies_between_different_faces(self) -> None:
        model = VisualModel()
        a = model.infer_face(_face(image=_gradient_face(0)))
        b = model.infer_face(_face(image=_gradient_face(7)))
        assert a != b
        assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0

    def test_visual_score_without_pixels_is_still_deterministic(self) -> None:
        model = VisualModel()
        assert model.infer_face(_face()) == model.infer_face(_face())

    def test_high_frequency_feature_is_pure(self) -> None:
        image = _gradient_face()
        assert face_high_frequency_ratio(image) == face_high_frequency_ratio(image)
        assert face_high_frequency_ratio(None) is None

    def test_audio_score_is_stable_for_the_same_spectrogram(self) -> None:
        rng = np.random.default_rng(1234)
        data = rng.standard_normal((128, 200)).astype("float32")
        spec = SpectrogramRepresentation(
            sample_rate=22050, n_mels=128, time_steps=200, data=data
        )
        model = AudioModel()
        scores = [model.infer([spec]) for _ in range(5)]
        assert len(set(scores)) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_audio_score_without_payload_is_still_deterministic(self) -> None:
        spec = SpectrogramRepresentation(
            sample_rate=22050, n_mels=128, time_steps=200, data=None
        )
        model = AudioModel()
        assert model.infer([spec]) == model.infer([spec])

    def test_likelihood_curve_is_centred_and_monotone(self) -> None:
        assert content_likelihood(0.5, 0.5, 0.1) == pytest.approx(0.5)
        low = content_likelihood(0.3, 0.5, 0.1)
        high = content_likelihood(0.7, 0.5, 0.1)
        assert low < 0.5 < high
        assert 0.0 < low < high < 1.0


class TestFrameExtractionBounds:
    def _decoded(self, count: int = 300) -> InMemoryDecodedVideo:
        frames = [
            Frame(index=i, timestamp=i / 30.0, width=64, height=64, image=None)
            for i in range(count)
        ]
        return InMemoryDecodedVideo(duration_seconds=count / 30.0, frame_rate=30.0, frames=frames)

    def test_max_frames_bounds_the_sample(self) -> None:
        extractor = FrameExtractor(decoder=lambda _p: self._decoded())
        frames = list(extractor.extract_frames("video.mp4", min_fps=1.0, max_frames=4))
        assert len(frames) == 4
        # Sampling still honours the >= 1 fps rule (every 30th native frame).
        assert [f.index for f in frames] == [0, 30, 60, 90]

    def test_no_cap_keeps_every_sampled_frame(self) -> None:
        extractor = FrameExtractor(decoder=lambda _p: self._decoded(90))
        frames = list(extractor.extract_frames("video.mp4", min_fps=1.0))
        assert len(frames) == 3

    def test_faces_are_in_a_stable_order(self) -> None:
        # Boxes are 200x200 on a 640x480 frame (>= the 5% face-area threshold).
        candidates = [
            FaceCandidate(x=50, y=60, width=200, height=200),
            FaceCandidate(x=10, y=60, width=200, height=200),
            FaceCandidate(x=30, y=10, width=200, height=200),
        ]
        frame = Frame(index=0, timestamp=0.0, width=640, height=480, image=None)
        extractor = FrameExtractor(detector=lambda _f: list(candidates))
        regions = extractor.isolate_faces(frame)
        assert [(r.y, r.x) for r in regions] == [(10, 30), (60, 10), (60, 50)]
        # Faces below the 5% area threshold are dropped as before.
        small = [FaceCandidate(x=0, y=0, width=40, height=40)]
        assert FrameExtractor(detector=lambda _f: small).isolate_faces(frame) == []


class TestDecoderIsLazy:
    def test_default_decoder_streams_frames(self) -> None:
        """The real decoder must not materialize the whole video."""
        import inspect

        from detection.processing.frame_extractor import _OpenCVDecodedVideo

        source = inspect.getsource(_OpenCVDecodedVideo.raw_frames)
        assert "yield" in source
        # No eager list() of the whole stream.
        assert "list(" not in source

    def test_undecodable_video_raises_extraction_error(self) -> None:
        import tempfile
        from pathlib import Path

        from detection.processing.frame_extractor import _default_decoder

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.mp4"
            bad.write_bytes(b"not a video")
            with pytest.raises(ExtractionError):
                _default_decoder(str(bad))
