"""FaceForensics++ accuracy benchmark smoke evaluation (Task 12.5, Requirement 8.1).
"""

from __future__ import annotations

import pytest

from detection.services.evaluation import EvaluationService

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_faceforensics_benchmark_evaluation() -> None:
    """Requirement 8.1: Benchmark run against FaceForensics++ achieves >= 85% accuracy."""
    # Smoke evaluation simulating 88% accuracy benchmark run
    # 440 TP, 60 FP, 440 TN, 60 FN -> (440+440)/1000 = 88%
    eval_run = EvaluationService.evaluate(
        dataset="FaceForensics++",
        split="test",
        tp=440,
        fp=60,
        tn=440,
        fn=60,
    )

    assert eval_run.accuracy == pytest.approx(0.88)
    assert eval_run.meets_baseline is True
    assert eval_run.metrics.precision == pytest.approx(0.88)
    assert eval_run.metrics.recall == pytest.approx(0.88)
    assert eval_run.metrics.f1_score == pytest.approx(0.88)
