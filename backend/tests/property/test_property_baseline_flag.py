"""Property 19: Baseline flag reflects the 85% threshold (Task 12.4, Requirements 8.1, 8.4).

Feature: deepfake-detection-platform, Property 19: Baseline flag reflects the 85% threshold
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.evaluation import EvaluationService

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(
    tp=st.integers(min_value=0, max_value=100),
    fp=st.integers(min_value=0, max_value=100),
    tn=st.integers(min_value=0, max_value=100),
    fn=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_property_baseline_flag_85_percent(tp: int, fp: int, tn: int, fn: int) -> None:
    """Property 19: meets_baseline is True iff accuracy >= 0.85."""
    if tp + fp + tn + fn == 0:
        return

    eval_run = EvaluationService.evaluate("Benchmark", "test", tp, fp, tn, fn)

    if eval_run.accuracy >= 0.85:
        assert eval_run.meets_baseline is True
    else:
        assert eval_run.meets_baseline is False
