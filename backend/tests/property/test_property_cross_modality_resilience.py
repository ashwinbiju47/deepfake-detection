"""Property 6: A missing or failed modality does not abort the other modality (Task 9.6, Requirements 2.4, 3.4, 3.5, 3.6).

Feature: deepfake-detection-platform, Property 6: A missing or failed modality does not abort the other modality
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.fusion import FusionEngine

pytestmark = pytest.mark.property


@given(
    v_signal=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    a_signal=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)
@settings(max_examples=100)
def test_property_cross_modality_resilience(
    v_signal: float | None, a_signal: float | None
) -> None:
    """Property 6: Single-modality success still produces classification unless both missing."""
    fusion = FusionEngine.fuse(visual_likelihood=v_signal, audio_likelihood=a_signal)

    if v_signal is not None or a_signal is not None:
        assert fusion.inconclusive is False
        assert fusion.score is not None
        assert fusion.label in ("authentic", "deepfake")
    else:
        assert fusion.inconclusive is True
        assert fusion.score is None
        assert fusion.label is None
