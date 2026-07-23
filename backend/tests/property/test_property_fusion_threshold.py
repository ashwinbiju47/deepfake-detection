"""Property 9: Classification label matches the decision threshold (Task 7.3, Requirement 4.3).

Feature: deepfake-detection-platform, Property 9: Classification label matches the decision threshold
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.fusion import FusionEngine

pytestmark = pytest.mark.property


@given(
    score_val=st.floats(min_value=0.0, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_property_classification_label_matches_threshold(
    score_val: float, threshold: float
) -> None:
    """Property 9: Label is 'deepfake' iff score >= threshold, else 'authentic'."""
    res = FusionEngine.fuse(visual_likelihood=score_val, threshold=threshold)

    assert res.score is not None
    if res.score >= threshold:
        assert res.label == "deepfake"
    else:
        assert res.label == "authentic"
