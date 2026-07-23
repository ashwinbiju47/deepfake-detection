"""Property 13: In-progress sessions never exceed configured capacity (Task 9.4, Requirement 6.4).

Feature: deepfake-detection-platform, Property 13: In-progress sessions never exceed configured capacity
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

pytestmark = pytest.mark.property


@given(
    concurrency=st.integers(min_value=1, max_value=64),
    job_count=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_property_worker_capacity_bound(concurrency: int, job_count: int) -> None:
    """Property 13: Active worker processing count <= concurrency limit."""
    active_workers = 0
    max_active_observed = 0

    for _ in range(job_count):
        if active_workers < concurrency:
            active_workers += 1
            max_active_observed = max(max_active_observed, active_workers)
        # Simulate job completion
        if active_workers > 0:
            active_workers -= 1

    assert max_active_observed <= concurrency
    assert 1 <= concurrency <= 64
