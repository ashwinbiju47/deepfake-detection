"""Property 8: Fusion is deterministic, range-bounded, and modality-aware (Task 7.2, Requirements 4.1, 4.2, 4.4).

Feature: deepfake-detection-platform, Property 8: Fusion is deterministic, range-bounded, and modality-aware
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.fusion import FusionEngine

pytestmark = pytest.mark.property


@given(
    v=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    a=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    w_v=st.floats(min_value=0.1, max_value=2.0),
    w_a=st.floats(min_value=0.1, max_value=2.0),
    thresh=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_property_fusion_determinism_and_modality_awareness(
    v: float | None, a: float | None, w_v: float, w_a: float, thresh: float
) -> None:
    """Property 8: Multi-modal fusion is deterministic, bounded, and modality-aware."""
    res1 = FusionEngine.fuse(v, a, weight_visual=w_v, weight_audio=w_a, threshold=thresh)
    res2 = FusionEngine.fuse(v, a, weight_visual=w_v, weight_audio=w_a, threshold=thresh)

    # Determinism
    assert res1 == res2

    # Inconclusive when both missing
    if v is None and a is None:
        assert res1.inconclusive is True
        assert res1.score is None
        assert res1.label is None
        assert res1.modalities_used == []
    else:
        assert res1.inconclusive is False
        assert res1.score is not None
        assert 0.0 <= res1.score <= 1.0
        assert res1.label in ("authentic", "deepfake")

        if v is not None and a is not None:
            assert res1.modalities_used == ["visual", "audio"]
        elif v is not None:
            assert res1.modalities_used == ["visual"]
            assert res1.score == pytest.approx(v)
        else:
            assert res1.modalities_used == ["audio"]
            assert res1.score == pytest.approx(a)
