"""Property 20: Grad-CAM heatmaps are normalized to [0.0, 1.0] (Task 15.2, Requirement 11.1).

Feature: deepfake-detection-platform, Property 20: Grad-CAM heatmaps are normalized to [0.0, 1.0]
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.xai import normalize_heatmap

pytestmark = pytest.mark.property


@given(
    matrix=st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_property_heatmap_normalization_bounds(matrix: list[list[float]]) -> None:
    """Property 20: Heatmap normalization scales all values into [0.0, 1.0]."""
    norm = normalize_heatmap(matrix)

    assert len(norm) == len(matrix)
    for row in norm:
        for val in row:
            assert 0.0 <= val <= 1.0
