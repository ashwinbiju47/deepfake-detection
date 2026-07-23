"""Model evaluation and metrics persistence (Task 12.1, Requirement 8).

This module implements the design's ``Model_Evaluator`` interface:

    evaluate(dataset, split, tp, fp, tn, fn) -> ModelEvaluation

Design intent (design.md -> Components and Interfaces -> Model_Evaluator):
* Compute confusion-matrix-derived accuracy, precision, recall, and F1 score (Requirement 8.2).
* Persist ModelEvaluation and EvaluationMetrics records in the database (Requirement 8.3).
* Set meets_baseline=True iff accuracy >= 0.85 (Requirement 8.1, 8.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from django.db import transaction

from detection.models import EvaluationMetrics, ModelEvaluation


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        t = self.total
        if t <= 0:
            return 0.0
        return (self.tp + self.tn) / float(t)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        if denom <= 0:
            return 0.0
        return self.tp / float(denom)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        if denom <= 0:
            return 0.0
        return self.tp / float(denom)

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        if p + r <= 0:
            return 0.0
        return 2.0 * (p * r) / (p + r)

    def to_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


class EvaluationService:
    """Computes ML benchmark metrics and persists evaluation runs."""

    @staticmethod
    def evaluate(
        dataset: str,
        split: str,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> ModelEvaluation:
        """Compute metrics, check 85% baseline threshold, and persist Evaluation records."""
        cm = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
        acc = cm.accuracy
        prec = cm.precision
        rec = cm.recall
        f1 = cm.f1_score

        meets_baseline = acc >= 0.85

        with transaction.atomic():
            eval_run = ModelEvaluation.objects.create(
                dataset=dataset,
                split=split,
                accuracy=acc,
                meets_baseline=meets_baseline,
            )

            EvaluationMetrics.objects.create(
                run=eval_run,
                confusion_matrix=cm.to_dict(),
                precision=prec,
                recall=rec,
                f1_score=f1,
            )


        return eval_run
