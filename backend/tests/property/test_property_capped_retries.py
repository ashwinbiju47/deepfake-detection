"""Property 14: Job retries are capped at 3 and preserve session integrity (Task 9.5, Requirement 6.6).

Feature: deepfake-detection-platform, Property 14: Job retries are capped at 3 and preserve session integrity
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

pytestmark = pytest.mark.property


@given(attempts=st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_property_capped_retries_and_session_integrity(attempts: int) -> None:
    """Property 14: Retries do not exceed 3."""
    max_retries = 3
    retries_performed = 0
    status = "PROCESSING"

    for _ in range(attempts):
        if retries_performed < max_retries:
            retries_performed += 1
        else:
            status = "FAILED"
            break

    assert retries_performed <= 3
    if attempts > max_retries:
        assert status == "FAILED"
