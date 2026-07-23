"""Property 18: Evaluation metrics persist and round-trip (Task 12.3, Requirement 8.3).

Feature: deepfake-detection-platform, Property 18: Evaluation metrics persist and round-trip
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.models import ModelEvaluation
from detection.services.evaluation import EvaluationService

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(
    dataset=st.sampled_from(["FaceForensics++", "DFDC", "Celeb-DF"]),
    split=st.sampled_from(["train", "val", "test"]),
    tp=st.integers(min_value=1, max_value=500),
    fp=st.integers(min_value=0, max_value=500),
    tn=st.integers(min_value=1, max_value=500),
    fn=st.integers(min_value=0, max_value=500),
)
@settings(max_examples=100)
def test_property_metrics_persistence_roundtrip(
    dataset: str, split: str, tp: int, fp: int, tn: int, fn: int
) -> None:
    """Property 18: Evaluation runs persist metrics and round-trip from DB accurately."""
    eval_run = EvaluationService.evaluate(dataset, split, tp, fp, tn, fn)

    fetched = ModelEvaluation.objects.get(run_id=eval_run.run_id)
    assert fetched.dataset == dataset
    assert fetched.split == split
    assert fetched.accuracy == pytest.approx(eval_run.accuracy)
    assert fetched.metrics.precision == pytest.approx(eval_run.metrics.precision)
    assert fetched.metrics.recall == pytest.approx(eval_run.metrics.recall)
    assert fetched.metrics.f1_score == pytest.approx(eval_run.metrics.f1_score)
    assert fetched.metrics.confusion_matrix == {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
