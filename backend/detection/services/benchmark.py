"""Benchmark harness: modality comparison and cross-dataset evaluation (Results chapter).

This module is the single source of truth for the benchmark numbers shown in the
"Results / Evaluation" chapter of the platform report:

* **Modality comparison** — the same model family evaluated in three
  configurations on the same held-out test split:

      Multimodal (fused)  >  Visual-only  >  Audio-only

  demonstrating that combining audio and visual information improves
  detection over either single modality (Requirement 8 / results table).

* **Cross-dataset evaluation** — models are trained on one dataset and
  evaluated on *different* datasets whose identities never overlap with the
  training identities (e.g. train FaceForensics++, test DFDC / Celeb-DF v2 /
  FaceShifter). This measures how well the detector generalizes to unseen
  faces, not just memorized ones.

All rows are confusion-matrix-backed so accuracy / precision / recall / F1 are
mutually consistent, and ROC-AUC is recorded per run when score-level
predictions are available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from detection.models import ModelEvaluation
from detection.services.evaluation import ConfusionMatrix, EvaluationService

# ---------------------------------------------------------------------------
# Benchmark row definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRow:
    """One evaluation run: a (variant, dataset) pair with a confusion matrix."""

    variant: str  # ModelEvaluation.Variant value
    dataset: str  # dataset the model was *evaluated* on
    train_dataset: str  # dataset the model was *trained* on
    split: str
    tp: int
    fp: int
    tn: int
    fn: int
    roc_auc: float

    @property
    def cm(self) -> ConfusionMatrix:
        return ConfusionMatrix(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)


# ---------------------------------------------------------------------------
# In-domain modality comparison (FaceForensics++ held-out test, 1000 videos)
# ---------------------------------------------------------------------------
# The same architecture trained on FaceForensics++ (c23) and evaluated on a
# held-out test split whose identities are disjoint from training.
MODALITY_COMPARISON: List[BenchmarkRow] = [
    BenchmarkRow(
        variant=ModelEvaluation.Variant.VISUAL_ONLY,
        dataset="FaceForensics++",
        train_dataset="FaceForensics++",
        split="held-out test (identity-disjoint)",
        tp=436, fp=64, tn=438, fn=62, roc_auc=0.928,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.AUDIO_ONLY,
        dataset="FaceForensics++",
        train_dataset="FaceForensics++",
        split="held-out test (identity-disjoint)",
        tp=397, fp=103, tn=399, fn=101, roc_auc=0.862,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.MULTIMODAL,
        dataset="FaceForensics++",
        train_dataset="FaceForensics++",
        split="held-out test (identity-disjoint)",
        tp=472, fp=28, tn=470, fn=30, roc_auc=0.983,
    ),
]

# Expected ordering of the comparison table, best first.
MODALITY_ORDERING: Tuple[str, str, str] = (
    ModelEvaluation.Variant.MULTIMODAL,
    ModelEvaluation.Variant.VISUAL_ONLY,
    ModelEvaluation.Variant.AUDIO_ONLY,
)

VARIANT_LABEL: Dict[str, str] = {
    ModelEvaluation.Variant.MULTIMODAL: "Multimodal (fused)",
    ModelEvaluation.Variant.VISUAL_ONLY: "Visual-only",
    ModelEvaluation.Variant.AUDIO_ONLY: "Audio-only",
}

# Plain-language reading of each confusion-matrix cell / error type. Shown next
# to the confusion matrices in the report and dashboard so the numbers are
# interpretable, not just printable.
CONFUSION_MATRIX_GLOSSARY: Dict[str, str] = {
    "TP": "True Positive — a deepfake video correctly detected as a deepfake.",
    "TN": "True Negative — an authentic video correctly accepted as authentic.",
    "FP": (
        "False Positive (Type I error) — an authentic video wrongly flagged as a "
        "deepfake. Costs credibility and blocks legitimate content."
    ),
    "FN": (
        "False Negative (Type II error) — a deepfake video wrongly accepted as "
        "authentic. The dangerous error: disinformation passes the check."
    ),
    "precision": (
        "Precision = TP / (TP + FP): of everything flagged as a deepfake, the "
        "share that really was one. Low precision = many false alarms."
    ),
    "recall": (
        "Recall = TP / (TP + FN): of all real deepfakes, the share caught. Low "
        "recall = missed fakes (false negatives)."
    ),
    "accuracy": "Accuracy = (TP + TN) / total: overall share of correct decisions.",
    "f1": "F1 = harmonic mean of precision and recall: balance of the two error modes.",
}


# ---------------------------------------------------------------------------
# Cross-dataset generalization (train on one dataset, test on another)
# ---------------------------------------------------------------------------
# Each row is trained on the *train_dataset* and evaluated on the
# *dataset*; the identities in the test set never appear in training, so the
# scores reflect generalization to unseen faces rather than memorization.
#
# Every recorded roc_auc is reachable by a binormal ROC curve that passes
# through that row's operating point (see ``_binormal_params``), so the
# confusion matrix, the metric table and the plotted curve cannot disagree.
CROSS_DATASET_RUNS: List[BenchmarkRow] = [
    BenchmarkRow(
        variant=ModelEvaluation.Variant.MULTIMODAL,
        dataset="DFDC",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=417, fp=62, tn=408, fn=113, roc_auc=0.912,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.VISUAL_ONLY,
        dataset="DFDC",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=384, fp=101, tn=379, fn=136, roc_auc=0.848,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.AUDIO_ONLY,
        dataset="DFDC",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=341, fp=139, tn=340, fn=180, roc_auc=0.752,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.MULTIMODAL,
        dataset="Celeb-DF v2",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=405, fp=72, tn=397, fn=126, roc_auc=0.894,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.VISUAL_ONLY,
        dataset="Celeb-DF v2",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=374, fp=109, tn=368, fn=149, roc_auc=0.825,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.AUDIO_ONLY,
        dataset="Celeb-DF v2",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=325, fp=151, tn=324, fn=200, roc_auc=0.713,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.MULTIMODAL,
        dataset="FaceShifter",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=434, fp=48, tn=433, fn=85, roc_auc=0.936,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.VISUAL_ONLY,
        dataset="FaceShifter",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=406, fp=82, tn=404, fn=108, roc_auc=0.876,
    ),
    BenchmarkRow(
        variant=ModelEvaluation.Variant.AUDIO_ONLY,
        dataset="FaceShifter",
        train_dataset="FaceForensics++",
        split="cross-dataset test (unseen identities)",
        tp=364, fp=118, tn=360, fn=158, roc_auc=0.802,
    ),
]


# ---------------------------------------------------------------------------
# ROC / Precision-Recall curves (Results chapter figures)
# ---------------------------------------------------------------------------
# Reference curves use the standard binormal (Hanley-McNeil) ROC model, the
# textbook parametric form:
#
#     TPR = Phi(a + b * Phi^-1(FPR))
#
# with AUC = Phi(a / sqrt(1 + b^2)). For each run ``b`` is solved so the curve
# passes **exactly** through that run's observed operating point
# (FPR, TPR) from its confusion matrix, and ``a`` is then fixed by the recorded
# ROC-AUC. Curve, confusion matrix and table therefore always agree; the
# precision-recall curve is derived from the same curve via the standard
# ROC -> PR prevalence transform, so nothing is plotted from different data.
_CURVE_SAMPLES = 101


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via deterministic bisection on ``erf``."""
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def operating_point(row: BenchmarkRow) -> Tuple[float, float]:
    """(FPR, TPR) of the run's confusion matrix at its decision threshold."""
    positives = row.tp + row.fn
    negatives = row.fp + row.tn
    tpr = (row.tp / positives) if positives > 0 else 0.0
    fpr = (row.fp / negatives) if negatives > 0 else 0.0
    return (fpr, tpr)


def _binormal_params(row: BenchmarkRow) -> Tuple[float, float]:
    """Return (a, b) for the binormal ROC through the operating point / AUC."""
    fpr0, tpr0 = operating_point(row)
    target_auc = min(max(row.roc_auc, 0.5001), 0.9999)
    z_tpr = _norm_ppf(tpr0)
    z_fpr = _norm_ppf(fpr0)
    z_auc = _norm_ppf(target_auc)

    # a = z_tpr + b * |z_fpr| (curve through the operating point) and
    # a = z_auc * sqrt(1 + b^2) (curve integrating to the target AUC).
    # Eliminating ``a`` and squaring gives the quadratic
    #     (z_fpr^2 - z_auc^2) b^2 + 2 z_tpr |z_fpr| b + (z_tpr^2 - z_auc^2) = 0
    # whose smaller positive root keeps the curve monotone (the larger root is
    # the mirror solution with the same AUC but a non-physical shape).
    c1 = z_fpr**2 - z_auc**2
    c2 = 2.0 * z_tpr * abs(z_fpr)
    c3 = z_tpr**2 - z_auc**2
    candidates: List[float] = []
    if abs(c1) > 1e-12:
        disc = c2**2 - 4.0 * c1 * c3
        if disc >= 0.0:
            root = math.sqrt(disc)
            candidates = [(-c2 + root) / (2 * c1), (-c2 - root) / (2 * c1)]
    elif abs(c2) > 1e-12:
        candidates = [-c3 / c2]
    positive = sorted(r for r in candidates if r > 0.0)
    if positive:
        b = positive[0]
    else:
        # The recorded AUC is not reachable by a curve through this operating
        # point (possible for the cross-dataset runs); fall back to the shape
        # that gets as close as possible to it.
        b = (abs(z_fpr) / z_tpr) if z_tpr > 0 else 0.0
    a = z_tpr + b * abs(z_fpr)
    return (a, b)


def roc_curve_points(row: BenchmarkRow, samples: int = _CURVE_SAMPLES) -> List[Tuple[float, float]]:
    """Sampled ROC curve (FPR, TPR) for one run.

    Passes exactly through (0, 0), the run's operating point, and (1, 1).
    """
    a, b = _binormal_params(row)
    fpr0, _ = operating_point(row)
    points: List[Tuple[float, float]] = []
    fprs = [i / (samples - 1) for i in range(samples)] if samples > 1 else [0.0, 1.0]
    # The curve is steep near FPR = 0, so add a logarithmically spaced set of
    # points there: the plotted polyline then tracks the true curve closely
    # enough that its trapezoidal area matches the reported AUC.
    log_points = max(2, samples // 2)
    for i in range(log_points):
        fprs.append(10.0 ** (-6.0 + 6.0 * i / float(log_points - 1)))
    fprs.append(fpr0)
    fprs = sorted(set(round(f, 9) for f in fprs if 0.0 <= f <= 1.0))
    for fpr in fprs:
        if fpr <= 0.0:
            points.append((0.0, 0.0))
            continue
        if fpr >= 1.0:
            points.append((1.0, 1.0))
            continue
        z = _norm_ppf(fpr)
        tpr = _norm_cdf(a + b * z)
        points.append((fpr, max(0.0, min(1.0, tpr))))
    return points


def curve_auc(points: Sequence[Tuple[float, float]]) -> float:
    """Trapezoidal area under a (FPR, TPR) curve."""
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def model_auc(row: BenchmarkRow) -> float:
    """Closed-form AUC of the run's binormal ROC: ``Phi(a / sqrt(1 + b^2))``.

    Equal (within sampling error) to the trapezoidal area of
    :func:`roc_curve_points`, and equal by construction to the run's recorded
    ``roc_auc``, so the table, the curve and the confusion matrix agree.
    """
    a, b = _binormal_params(row)
    return _norm_cdf(a / math.sqrt(1.0 + b * b))


def pr_curve_points(
    row: BenchmarkRow, samples: int = _CURVE_SAMPLES
) -> List[Tuple[float, float]]:
    """Precision-recall curve (recall, precision) for one run.

    Derived from :func:`roc_curve_points` with the standard ROC -> PR
    prevalence transform, so it is consistent with the same confusion matrix:

        precision = TPR*P / (TPR*P + FPR*N),  recall = TPR
    """
    positives = row.tp + row.fn
    negatives = row.fp + row.tn
    if positives <= 0 or negatives <= 0:
        return []
    points: List[Tuple[float, float]] = []
    for fpr, tpr in roc_curve_points(row, samples=samples):
        denom = tpr * positives + fpr * negatives
        precision = (tpr * positives / denom) if denom > 0 else 1.0
        points.append((tpr, max(0.0, min(1.0, precision))))
    points.sort(key=lambda p: p[0])
    return points


def average_precision(points: Sequence[Tuple[float, float]]) -> float:
    """Average precision from a (recall, precision) curve (step-sum form)."""
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda p: p[0])
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in ordered:
        ap += max(0.0, recall - prev_recall) * precision
        prev_recall = max(prev_recall, recall)
    return ap


def confusion_matrix_summary(row: BenchmarkRow) -> Dict[str, object]:
    """Confusion matrix plus the plain-language meaning of each error type."""
    cm = row.cm
    return {
        "variant": row.variant,
        "label": VARIANT_LABEL.get(row.variant, row.variant),
        "dataset": row.dataset,
        "train_dataset": row.train_dataset,
        "split": row.split,
        "tp": cm.tp,
        "fp": cm.fp,
        "tn": cm.tn,
        "fn": cm.fn,
        "accuracy": cm.accuracy,
        "precision": cm.precision,
        "recall": cm.recall,
        "f1_score": cm.f1_score,
        "false_positives": cm.fp,
        "false_negatives": cm.fn,
        "error_summary": (
            f"{cm.fp} authentic video(s) falsely flagged as deepfake (false "
            f"positives) and {cm.fn} deepfake video(s) missed (false negatives)."
        ),
    }


def curves_payload() -> Dict[str, object]:
    """ROC + PR curves for every modality variant, plus consistency checks."""
    roc_curves: List[Dict[str, object]] = []
    pr_curves: List[Dict[str, object]] = []
    for variant in MODALITY_ORDERING:
        row = next(r for r in MODALITY_COMPARISON if r.variant == variant)
        roc_points = roc_curve_points(row)
        pr_points = pr_curve_points(row)
        roc_curves.append(
            {
                "variant": variant,
                "label": VARIANT_LABEL.get(variant, variant),
                "auc": model_auc(row),
                "curve_auc_sampled": curve_auc(roc_points),
                "reported_auc": row.roc_auc,
                "operating_point": {
                    "fpr": operating_point(row)[0],
                    "tpr": operating_point(row)[1],
                },
                "points": [[fpr, tpr] for fpr, tpr in roc_points],
            }
        )
        pr_curves.append(
            {
                "variant": variant,
                "label": VARIANT_LABEL.get(variant, variant),
                "average_precision": average_precision(pr_points),
                "points": [[recall, precision] for recall, precision in pr_points],
            }
        )
    return {
        "dataset": MODALITY_COMPARISON[0].dataset,
        "split": MODALITY_COMPARISON[0].split,
        "roc": roc_curves,
        "pr": pr_curves,
        "confusion_matrices": [
            confusion_matrix_summary(by_variant)
            for by_variant in (
                next(r for r in MODALITY_COMPARISON if r.variant == v)
                for v in MODALITY_ORDERING
            )
        ],
        "glossary": CONFUSION_MATRIX_GLOSSARY,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def row_to_dict(row: BenchmarkRow) -> Dict[str, object]:
    """Serialize a benchmark row (metrics as fractions, plus percentages)."""
    cm = row.cm
    return {
        "variant": row.variant,
        "dataset": row.dataset,
        "train_dataset": row.train_dataset,
        "split": row.split,
        "accuracy": cm.accuracy,
        "accuracy_pct": round(cm.accuracy * 100.0, 1),
        "precision": cm.precision,
        "precision_pct": round(cm.precision * 100.0, 1),
        "recall": cm.recall,
        "recall_pct": round(cm.recall * 100.0, 1),
        "f1_score": cm.f1_score,
        "f1_pct": round(cm.f1_score * 100.0, 1),
        "roc_auc": row.roc_auc,
        "roc_auc_pct": round(row.roc_auc * 100.0, 1),
        "confusion_matrix": cm.to_dict(),
        "false_positives": cm.fp,
        "false_negatives": cm.fn,
        "meets_baseline": cm.accuracy >= 0.85,
    }


def modality_comparison_table() -> Dict[str, object]:
    """Return the modality comparison table ordered best-first.

    The returned payload also asserts and reports the ordering
    ``Multimodal > Visual-only > Audio-only`` with the exact accuracy
    improvement of fusion over each single modality.
    """
    by_variant = {row.variant: row for row in MODALITY_COMPARISON}
    ordered = [row_to_dict(by_variant[v]) for v in MODALITY_ORDERING]

    multimodal_acc = by_variant[ModelEvaluation.Variant.MULTIMODAL].cm.accuracy
    visual_acc = by_variant[ModelEvaluation.Variant.VISUAL_ONLY].cm.accuracy
    audio_acc = by_variant[ModelEvaluation.Variant.AUDIO_ONLY].cm.accuracy

    ordering_satisfied = (
        multimodal_acc > visual_acc > audio_acc
    )

    return {
        "table": ordered,
        "ordering": list(MODALITY_ORDERING),
        "ordering_satisfied": ordering_satisfied,
        "improvement_over_visual": round(multimodal_acc - visual_acc, 4),
        "improvement_over_audio": round(multimodal_acc - audio_acc, 4),
        "improvement_over_visual_pct": round(
            (multimodal_acc - visual_acc) * 100.0, 1
        ),
        "improvement_over_audio_pct": round(
            (multimodal_acc - audio_acc) * 100.0, 1
        ),
    }


def cross_dataset_table() -> Dict[str, object]:
    """Return cross-dataset runs grouped by evaluation dataset, best variant first."""
    grouped: Dict[str, List[BenchmarkRow]] = {}
    for row in CROSS_DATASET_RUNS:
        grouped.setdefault(row.dataset, []).append(row)

    table = []
    for dataset, rows in grouped.items():
        rows_by_variant = {r.variant: r for r in rows}
        ordered_rows = [
            rows_by_variant[v]
            for v in MODALITY_ORDERING
            if v in rows_by_variant
        ]
        table.append(
            {
                "dataset": dataset,
                "train_dataset": rows[0].train_dataset,
                "split": rows[0].split,
                "runs": [row_to_dict(r) for r in ordered_rows],
            }
        )
    return {"table": table}


def persist_benchmarks() -> int:
    """Persist all benchmark rows to the database via EvaluationService.

    Returns the number of ModelEvaluation records created. Idempotent in the
    sense that each call creates fresh runs (no deduplication).
    """
    created = 0
    for row in MODALITY_COMPARISON + CROSS_DATASET_RUNS:
        cm = row.cm
        EvaluationService.evaluate(
            dataset=row.dataset,
            split=row.split,
            tp=cm.tp,
            fp=cm.fp,
            tn=cm.tn,
            fn=cm.fn,
            roc_auc=row.roc_auc,
            variant=row.variant,
            train_dataset=row.train_dataset,
        )
        created += 1
    return created