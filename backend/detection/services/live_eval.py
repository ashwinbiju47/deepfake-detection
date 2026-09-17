"""Live evaluation from the platform's own labeled sessions (Results chapter).

The reference benchmark tables in :mod:`detection.services.benchmark` are
**published-dataset constants** (the FaceForensics++ held-out /
cross-dataset runs quoted in the report). This module computes the *live*
counterpart: ROC curves, precision-recall curves and confusion matrices built
from the sessions actually analyzed on this deployment, using each session's
optional user-declared ``ground_truth`` (``real`` / ``fake``).

How the three model variants are evaluated live
-----------------------------------------------
For every labeled, completed session there are up to two per-modality
likelihoods plus the fused score:

* **multimodal** — the fused score from ``FusionResult`` (what the platform
  actually reports to the user);
* **visual_only** — ``VisualResult.aggregate_likelihood``;
* **audio_only** — ``AudioResult.likelihood``.

Sessions missing the modality's likelihood are skipped *for that variant
only*, so e.g. an image upload contributes its visual evidence to the
visual-only and multimodal samples but not to audio-only.

The confusion matrix is read at the deployment's decision threshold, and the
ROC/PR curves are built from the same score samples (thresholds swept between
the observed min and max). With a small number of live samples the curves are
step functions — that is expected and rendered accordingly.

Nothing here mutates the reference constants: ``live=False`` payloads fall
back to the published-dataset tables, so the platform never fabricates a
live curve from data it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from django.db.models import Q

from detection.models import AnalysisSession
from detection.services.benchmark import CONFUSION_MATRIX_GLOSSARY, VARIANT_LABEL
from detection.services.evaluation import ConfusionMatrix

# Minimum labeled samples per variant before a curve is meaningful.
MIN_SAMPLES_FOR_CURVE = 2
# Curve resolution: sweep every distinct score (step curve), capped for sanity.
_MAX_CURVE_POINTS = 512

_VARIANT_ORDER = ("multimodal", "visual_only", "audio_only")


@dataclass(frozen=True)
class _Sample:
    """One labeled observation: a score, the true label, and the session id."""

    score: float
    positive: bool  # True = ground truth "fake"
    session_id: str


def collect_samples() -> Dict[str, List[_Sample]]:
    """Gather labeled score samples per model variant from completed sessions."""
    sessions = (
        AnalysisSession.objects.filter(
            status=AnalysisSession.Status.COMPLETED,
        )
        .filter(Q(ground_truth="real") | Q(ground_truth="fake"))
        .prefetch_related("visual_result", "audio_result", "fusion_result")
    )

    multimodal: List[_Sample] = []
    visual: List[_Sample] = []
    audio: List[_Sample] = []

    for session in sessions:
        positive = session.ground_truth == AnalysisSession.GroundTruth.FAKE
        fusion = getattr(session, "fusion_result", None)
        vres = getattr(session, "visual_result", None)
        ares = getattr(session, "audio_result", None)

        if fusion is not None and fusion.score is not None and not fusion.inconclusive:
            multimodal.append(_Sample(float(fusion.score), positive, str(session.id)))
        if vres is not None and vres.aggregate_likelihood is not None:
            visual.append(
                _Sample(float(vres.aggregate_likelihood), positive, str(session.id))
            )
        if ares is not None and ares.likelihood is not None:
            audio.append(_Sample(float(ares.likelihood), positive, str(session.id)))

    return {
        "multimodal": multimodal,
        "visual_only": visual,
        "audio_only": audio,
    }


def _roc_curve(samples: List[_Sample]) -> List[Tuple[float, float]]:
    """Step ROC curve (fpr, tpr) from score samples, swept threshold-style.

    Thresholds sweep from above the max score (everything predicted
    negative -> (0, 0)) to the min score (everything positive -> (1, 1)),
    exactly as an empirical ROC is built.
    """
    if not samples:
        return []
    thresholds = sorted({s.score for s in samples}, reverse=True)
    points: List[Tuple[float, float]] = [(0.0, 0.0)]
    total_pos = sum(1 for s in samples if s.positive)
    total_neg = len(samples) - total_pos
    for t in thresholds:
        tp = sum(1 for s in samples if s.positive and s.score >= t)
        fp = sum(1 for s in samples if not s.positive and s.score >= t)
        tpr = tp / total_pos if total_pos else 1.0
        fpr = fp / total_neg if total_neg else 0.0
        points.append((fpr, tpr))
    points.append((1.0, 1.0))
    return points


def _trapezoid_auc(points: List[Tuple[float, float]]) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def _average_precision(points: List[Tuple[float, float]]) -> float:
    """Step-wise average precision from (recall, precision) points.

    AP = Σ (r_i − r_{i−1}) · p_i over recall-ascending points with r_0 = 0 —
    the standard definition. Unlike a raw trapezoid over the step curve this
    stays correct for tiny samples (a single perfect point (1, 1) gives
    AP = 1.0, not 0).
    """
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda p: p[0])
    area = 0.0
    prev_recall = 0.0
    for recall, precision in ordered:
        if recall > prev_recall:
            area += (recall - prev_recall) * precision
            prev_recall = recall
    return area


def _pr_curve(samples: List[_Sample]) -> List[Tuple[float, float]]:
    """Step precision-recall curve (recall, precision) from score samples."""
    if not samples:
        return []
    thresholds = sorted({s.score for s in samples}, reverse=True)
    total_pos = sum(1 for s in samples if s.positive)
    points: List[Tuple[float, float]] = []
    for t in thresholds:
        tp = sum(1 for s in samples if s.positive and s.score >= t)
        fp = sum(1 for s in samples if not s.positive and s.score >= t)
        recall = tp / total_pos if total_pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        points.append((recall, precision))
    return points


def _confusion_matrix(samples: List[_Sample], threshold: float) -> ConfusionMatrix:
    tp = sum(1 for s in samples if s.positive and s.score >= threshold)
    fn = sum(1 for s in samples if s.positive and s.score < threshold)
    fp = sum(1 for s in samples if not s.positive and s.score >= threshold)
    tn = sum(1 for s in samples if not s.positive and s.score < threshold)
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def _variant_payload(variant: str, samples: List[_Sample]) -> Optional[Dict[str, object]]:
    """Curve + matrix payload for one variant, or None when too few samples."""
    if len(samples) < MIN_SAMPLES_FOR_CURVE:
        return None

    scores = [s.score for s in samples]
    low, high = min(scores), max(scores)
    roc = _roc_curve(samples)
    pr = _pr_curve(samples)

    # Curve points are capped: with many samples, subsample the sweep evenly.
    if len(roc) > _MAX_CURVE_POINTS:
        step = len(roc) / float(_MAX_CURVE_POINTS)
        roc = [roc[int(i * step)] for i in range(_MAX_CURVE_POINTS)]
    if len(pr) > _MAX_CURVE_POINTS:
        step = len(pr) / float(_MAX_CURVE_POINTS)
        pr = [pr[int(i * step)] for i in range(_MAX_CURVE_POINTS)]

    return {
        "variant": variant,
        "label": VARIANT_LABEL.get(variant, variant),
        "samples": len(samples),
        "score_range": {"min": low, "max": high},
        "roc": {
            "auc": _trapezoid_auc(roc),
            "points": [[fpr, tpr] for fpr, tpr in roc],
        },
        "pr": {
            "average_precision": _average_precision(pr),
            "points": [[recall, precision] for recall, precision in pr],
        },
    }


def live_evaluation_payload(threshold: Optional[float] = None) -> Dict[str, object]:
    """Build the live-evaluation section of the benchmark payload.

    Returns ``{"available": False, "reason": ...}`` when fewer than
    ``MIN_SAMPLES_FOR_CURVE`` labeled sessions exist — the frontend and the
    PDF report then show the reference figures with a note instead of an
    empty live panel.
    """
    samples_by_variant = collect_samples()
    total_labeled = sum(len(s) for s in samples_by_variant.values())

    if threshold is None:
        from django.conf import settings  # noqa: PLC0415

        threshold = float(getattr(settings, "DECISION_THRESHOLD", 0.5))

    variants: List[Dict[str, object]] = []
    matrices: List[Dict[str, object]] = []
    skipped: List[str] = []

    for variant in _VARIANT_ORDER:
        samples = samples_by_variant.get(variant, [])
        if len(samples) < MIN_SAMPLES_FOR_CURVE:
            skipped.append(variant)
            continue
        payload = _variant_payload(variant, samples)
        if payload is None:  # pragma: no cover - guarded above
            skipped.append(variant)
            continue
        variants.append(payload)

        cm = _confusion_matrix(samples, threshold)
        matrices.append(
            {
                "variant": variant,
                "label": VARIANT_LABEL.get(variant, variant),
                "samples": len(samples),
                "threshold": threshold,
                "tp": cm.tp,
                "fp": cm.fp,
                "tn": cm.tn,
                "fn": cm.fn,
                "accuracy": cm.accuracy,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1_score": cm.f1_score,
                "error_summary": (
                    f"{cm.fp} real media falsely flagged as deepfake (false "
                    f"positives) and {cm.fn} fake media missed (false negatives), "
                    f"across {len(samples)} labeled session(s) at threshold "                    f"{threshold:.2f}."),
            }
        )

    return {
        "available": bool(variants),
        "total_labeled_sessions": total_labeled,
        "reason": (
            None
            if variants
            else (
                "Live evaluation needs at least "
                f"{MIN_SAMPLES_FOR_CURVE} labeled sessions per model — label "
                "your analyses (Real / Fake) after they complete to build "
                "your own ROC curves and confusion matrices."
            )
        ),
        "min_samples": MIN_SAMPLES_FOR_CURVE,
        "variants": variants,
        "confusion_matrices": matrices,
        "glossary": CONFUSION_MATRIX_GLOSSARY,
    }
