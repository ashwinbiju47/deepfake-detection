"""Property 5: All likelihood values fall within [0.0, 1.0] (Task 6.3, Requirements 2.3, 2.6, 3.3).

Feature: deepfake-detection-platform, Property 5: All likelihood values fall within [0.0, 1.0]

For any visual or audio inference or aggregation, the resulting likelihood MUST
always fall strictly within the inclusive range [0.0, 1.0].
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.ml import AudioModel, VisualModel, clamp_likelihood
from detection.processing.audio_extractor import SpectrogramRepresentation
from detection.processing.frame_extractor import FaceRegion

pytestmark = pytest.mark.property


@given(raw_val=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_clamp_likelihood_bounds(raw_val: float) -> None:
    """Property 5: Likelihood values are strictly bounded in [0.0, 1.0]."""
    clamped = clamp_likelihood(raw_val)
    assert 0.0 <= clamped <= 1.0


@given(
    raw_face_val=st.floats(allow_nan=False, allow_infinity=False),
    width=st.integers(min_value=1, max_value=100),
    height=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_property_visual_model_infer_face_bounds(
    raw_face_val: float, width: int, height: int
) -> None:
    """Property 5: VisualModel.infer_face yields likelihood strictly in [0.0, 1.0]."""
    face = FaceRegion(
        frame_index=0, x=0, y=0, width=width, height=height, frame_width=100, frame_height=100
    )
    model = VisualModel(infer_backend=lambda _f: raw_face_val)
    score = model.infer_face(face)
    assert 0.0 <= score <= 1.0


@given(
    likelihoods=st.lists(st.floats(allow_nan=False, allow_infinity=False)),
    top_k=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
)
@settings(max_examples=100)
def test_property_visual_model_aggregate_bounds(
    likelihoods: list[float], top_k: int | None
) -> None:
    """Property 5: VisualModel.aggregate yields likelihood strictly in [0.0, 1.0]."""
    model = VisualModel()
    agg = model.aggregate(likelihoods, top_k=top_k)
    assert 0.0 <= agg <= 1.0


@given(raw_audio_val=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_property_audio_model_infer_bounds(raw_audio_val: float) -> None:
    """Property 5: AudioModel.infer yields likelihood strictly in [0.0, 1.0]."""
    spec = SpectrogramRepresentation(sample_rate=22050, n_mels=128, time_steps=100, data=None)
    model = AudioModel(infer_backend=lambda _s: raw_audio_val)
    score = model.infer([spec])
    assert 0.0 <= score <= 1.0
