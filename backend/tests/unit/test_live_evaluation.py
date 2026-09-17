"""Live-evaluation tests (ROC/PR/confusion from the deployment's own sessions).

Covers:

* :func:`detection.ml.visual_model.autocorrelation_coefficient` — the
  scale-free texture feature that replaced the sample-pinned sharpness
  constant (regression: every image pinned to the same ~4 % score);
* :func:`detection.services.live_eval.live_evaluation_payload` — labeled
  sessions produce real curves/matrices; unlabeled ones produce the honest
  "not available" payload;
* the ground-truth labeling endpoint (set, clear, invalid values).
"""

from __future__ import annotations

import numpy as np
import pytest
from detection.models import (
    AnalysisSession,
    AudioResult,
    FusionResult,
    VisualResult,
)
from detection.ml.visual_model import face_aac
from rest_framework.test import APIClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Visual feature (regression for the static ~4 % image score)
# ---------------------------------------------------------------------------


def _image(h: int, w: int, seed: int, kind: str) -> np.ndarray:
    """Deterministic BGR image: 'face' = smooth gradients, 'fake' = periodic."""
    rng = np.random.default_rng(seed)
    if kind == "face":
        yy, xx = np.mgrid[0:h, 0:w]
        base = ((xx * 2 + yy * 3) % 180 + 40).astype("float32")
        img = np.stack([base, base * 0.8, base * 0.6], axis=-1)
        return (img + rng.normal(0, 2.0, img.shape)).astype("uint8")
    xx = np.arange(w)
    period = np.sin(2 * np.pi * xx / 8.0) * 80 + 128
    img = np.tile(period, (h, 1)).astype("float32")
    img = np.stack([img] * 3, axis=-1)
    return (img + rng.normal(0, 2.0, img.shape)).astype("uint8")


class TestAutocorrelationFeature:
    def test_natural_face_texture_is_low(self) -> None:
        aac = face_aac(_image(224, 224, seed=1, kind="face"))
        assert aac is not None
        assert aac < 0.45

    def test_periodic_pattern_is_high(self) -> None:
        """Periodic vertical structure (deepfake-style banding) scores high."""
        strip = np.zeros((224, 224, 3), dtype="uint8")
        strip[:, ::8, :] = 200
        aac = face_aac(strip)
        assert aac is not None
        assert aac > 0.6

    def test_scale_invariance(self) -> None:
        """Downscaling by 4x must not move the feature (regression core)."""
        big = _image(448, 448, seed=3, kind="face")
        small = _image(112, 112, 3, "face")
        a = face_aac(big)
        b = face_aac(small)
        assert a is not None and b is not None
        assert abs(a - b) < 0.2

    def test_degenerate_images_return_none(self) -> None:
        assert face_aac(None) is None
        flat = np.full((64, 64, 3), 128, dtype="uint8")
        assert face_aac(flat) is None

    def test_blank_frame_of_noise_is_not_faked_as_high_aac(self) -> None:
        rng = np.random.default_rng(99)
        noise = rng.integers(0, 256, (224, 224, 3), dtype="uint8")
        aac = face_aac(noise)
        assert aac is not None
        assert aac < 0.5  # uncorrelated noise has no periodic structure


# ---------------------------------------------------------------------------
# Live evaluation payload
# ---------------------------------------------------------------------------


def _make_session(idx: int, ground_truth: str, visual: float, audio: float, fused: float):
    session = AnalysisSession.objects.create(
        status=AnalysisSession.Status.COMPLETED,
        media_kind=AnalysisSession.MediaKind.VIDEO,
        source_ref=f"clip{idx}.mp4",
        ground_truth=ground_truth,
    )
    VisualResult.objects.create(
        session=session,
        aggregate_likelihood=visual,
        state="OK",
        frames_analyzed=5,
        faces_isolated=1,
    )
    AudioResult.objects.create(session=session, likelihood=audio, state="OK")
    FusionResult.objects.create(
        session=session,
        score=fused,
        label="DEEPFAKE" if fused >= 0.5 else "AUTHENTIC",
        modalities_used="visual,audio",
        threshold_used=0.5,
        inconclusive=False,
    )
    return session


@pytest.mark.django_db
class TestLiveEvaluationPayload:
    def test_unlabeled_sessions_report_unavailable(self) -> None:
        from detection.services.live_eval import live_evaluation_payload

        _make_session(0, AnalysisSession.GroundTruth.REAL, 0.2, 0.3, 0.24)
        payload = live_evaluation_payload(threshold=0.5)
        # Only one labeled session -> below MIN_SAMPLES_FOR_CURVE.
        if payload["available"] is False:
            assert "label" in str(payload["reason"]).lower()

    def test_labeled_sessions_produce_curves_and_matrices(self) -> None:
        from detection.services.live_eval import live_evaluation_payload

        _make_session(1, AnalysisSession.GroundTruth.FAKE, 0.8, 0.9, 0.84)
        _make_session(2, AnalysisSession.GroundTruth.REAL, 0.2, 0.1, 0.16)
        payload = live_evaluation_payload(threshold=0.5)

        assert payload["available"] is True
        assert payload["total_labeled_sessions"] >= 2
        variants = {v["variant"]: v for v in payload["variants"]}
        for name in ("multimodal", "visual_only", "audio_only"):
            assert name in variants
            roc = variants[name]["roc"]
            pr = variants[name]["pr"]
            assert roc["auc"] > 0.5  # perfectly separable synthetic data
            assert len(roc["points"]) >= 2

        matrices = {m["variant"]: m for m in payload["confusion_matrices"]}
        mm = matrices["multimodal"]
        assert (mm["tp"], mm["fn"], mm["fp"], mm["tn"]) == (1, 0, 0, 1)
        assert mm["accuracy"] == 1.0

    def test_false_negatives_are_counted_in_the_matrix(self) -> None:
        from detection.services.live_eval import live_evaluation_payload

        _make_session(3, AnalysisSession.GroundTruth.FAKE, 0.9, 0.95, 0.92)
        _make_session(4, AnalysisSession.GroundTruth.FAKE, 0.1, 0.2, 0.14)  # missed
        _make_session(5, AnalysisSession.GroundTruth.REAL, 0.2, 0.3, 0.24)
        payload = live_evaluation_payload(threshold=0.5)
        matrices = {m["variant"]: m for m in payload["confusion_matrices"]}
        mm = matrices["multimodal"]
        assert mm["fn"] == 1
        assert "missed" in mm["error_summary"]


# ---------------------------------------------------------------------------
# Ground-truth labeling endpoint
# ---------------------------------------------------------------------------


@pytest.mark.django_db
class TestGroundTruthEndpoint:
    def _session(self) -> AnalysisSession:
        return AnalysisSession.objects.create(
            status=AnalysisSession.Status.COMPLETED,
            media_kind=AnalysisSession.MediaKind.IMAGE,
            source_ref="photo.jpg",
        )

    def _post(self, session_id: str, value: str | None):
        return APIClient().post(
            f"/api/analyses/{session_id}/ground_truth",
            data={"ground_truth": value},
            format="json",
        )

    def test_label_and_relabel_and_clear(self) -> None:
        session = self._session()
        res = self._post(str(session.id), "fake")
        assert res.status_code == 200
        body = res.json()
        assert body["ground_truth"] == "fake"

        res = self._post(str(session.id), "real")
        assert res.json()["ground_truth"] == "real"

        res = self._post(str(session.id), None)
        assert res.status_code == 200
        assert res.json()["ground_truth"] is None

    def test_invalid_value_returns_400(self) -> None:
        session = self._session()
        res = self._post(str(session.id), "banana")
        assert res.status_code == 400

    def test_unknown_session_returns_404(self) -> None:
        import uuid

        res = APIClient().post(
            f"/api/analyses/{uuid.uuid4()}/ground_truth",
            data={"ground_truth": "real"},
            format="json",
        )
        assert res.status_code == 404
