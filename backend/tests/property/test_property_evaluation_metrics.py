"""Property 17: Evaluation metrics satisfy their mathematical relationships (Task 12.2, Requirement 8.2).

Feature: deepfake-detection-platform, Property 17: Evaluation metrics satisfy their mathematical relationships
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.evaluation import ConfusionMatrix

pytestmark = pytest.mark.property


@given(
    tp=st.integers(min_value=0, max_value=1000),
    fp=st.integers(min_value=0, max_value=1000),
    tn=st.integers(min_value=0, max_value=1000),
    fn=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_evaluation_metrics_math(tp: int, fp: int, tn: int, fn: int) -> None:
    """Property 17: Accuracy, precision, recall, and F1 fall in [0.0, 1.0] and satisfy math definitions."""
    cm = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)

    acc = cm.accuracy
    prec = cm.precision
    rec = cm.recall
    f1 = cm.f1_score

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0

    if tp + tn + fp + fn > 0:
        assert acc == pytest.approx((tp + tn) / float(tp + tn + fp + fn))
    if tp + fp > 0:
        assert prec == pytest.approx(tp / float(tp + fp))
    if tp + fn > 0:
        assert rec == pytest.approx(tp / float(tp + fn))
    if prec + rec > 0:
        assert f1 == pytest.approx(2 * (prec * rec) / (prec + rec))
