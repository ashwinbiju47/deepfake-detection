"""Results-chapter figures: confusion matrices, ROC/PR curves, weight ablation.

Covers the evaluation analytics added on top of the benchmark harness:

* confusion matrices for multimodal / visual-only / audio-only with explicit
  false-positive / false-negative explanations,
* one ROC curve carrying all three models, and one precision-recall curve,
  both consistent with the metric table (curve AUC == reported ROC-AUC),
* the fusion-weight ablation study showing 0.6/0.4 is the swept optimum,
* the rendered PNG figures exposed through the benchmark endpoint.
"""

from __future__ import annotations

import base64

import pytest
from rest_framework.test import APIClient

from detection.models import ModelEvaluation
from detection.services.ablation import (
    ABLATION_ALPHAS,
    VALIDATION_SET,
    ablation_table,
    evaluate_alpha,
)
from detection.services.benchmark import (
    CONFUSION_MATRIX_GLOSSARY,
    MODALITY_COMPARISON,
    confusion_matrix_summary,
    curve_auc,
    curves_payload,
    model_auc,
    operating_point,
    roc_curve_points,
)
from detection.services.figures import (
    clear_cache,
    figure_png,
    figures_payload,
)

pytestmark = [pytest.mark.django_db, pytest.mark.unit]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Feature flags must default on (regression: "report disabled")
# ---------------------------------------------------------------------------


def test_heatmap_and_report_flags_default_on() -> None:
    """The XAI triple and PDF download work out of the box."""
    from django.conf import settings

    assert settings.HEATMAP_ENABLED is True
    assert settings.REPORT_ENABLED is True


def test_health_endpoint_reports_report_enabled() -> None:
    payload = APIClient().get("/api/health").json()
    assert payload["feature_flags"]["report_enabled"] is True
    assert payload["feature_flags"]["heatmap_enabled"] is True


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------


class TestConfusionMatrices:
    def test_one_matrix_per_modality_with_error_counts(self) -> None:
        summaries = curves_payload()["confusion_matrices"]
        assert [s["variant"] for s in summaries] == [
            ModelEvaluation.Variant.MULTIMODAL,
            ModelEvaluation.Variant.VISUAL_ONLY,
            ModelEvaluation.Variant.AUDIO_ONLY,
        ]
        for summary in summaries:
            assert summary["tp"] + summary["fp"] + summary["tn"] + summary["fn"] > 0
            assert summary["false_positives"] == summary["fp"]
            assert summary["false_negatives"] == summary["fn"]
            assert str(summary["fp"]) in summary["error_summary"]
            assert str(summary["fn"]) in summary["error_summary"]

    def test_multimodal_makes_fewer_errors_than_each_single_modality(self) -> None:
        by_variant = {
            s["variant"]: s for s in curves_payload()["confusion_matrices"]
        }
        multimodal = by_variant["multimodal"]
        for variant in ("visual_only", "audio_only"):
            other = by_variant[variant]
            assert multimodal["accuracy"] > other["accuracy"]
            assert multimodal["precision"] > other["precision"]
            assert multimodal["recall"] > other["recall"]

    def test_glossary_explains_both_error_types(self) -> None:
        assert "authentic" in CONFUSION_MATRIX_GLOSSARY["FP"]
        assert "deepfake" in CONFUSION_MATRIX_GLOSSARY["FP"]
        assert "deepfake" in CONFUSION_MATRIX_GLOSSARY["FN"]
        # False negative is called out as the costlier error for this domain.
        assert "dangerous" in CONFUSION_MATRIX_GLOSSARY["FN"].lower()
        assert "precision" in CONFUSION_MATRIX_GLOSSARY["precision"].lower()
        assert "recall" in CONFUSION_MATRIX_GLOSSARY["recall"].lower()

    def test_summary_matches_row_confusion_matrix(self) -> None:
        row = next(r for r in MODALITY_COMPARISON if r.variant == "multimodal")
        summary = confusion_matrix_summary(row)
        assert (summary["tp"], summary["fp"], summary["tn"], summary["fn"]) == (
            row.tp,
            row.fp,
            row.tn,
            row.fn,
        )


# ---------------------------------------------------------------------------
# ROC + Precision-Recall curves
# ---------------------------------------------------------------------------


class TestRocAndPrCurves:
    def test_one_roc_curve_covering_all_three_models(self) -> None:
        payload = curves_payload()
        assert [c["variant"] for c in payload["roc"]] == [
            "multimodal",
            "visual_only",
            "audio_only",
        ]
        assert [c["variant"] for c in payload["pr"]] == [
            "multimodal",
            "visual_only",
            "audio_only",
        ]

    def test_roc_curve_runs_from_origin_to_top_right(self) -> None:
        points = roc_curve_points(MODALITY_COMPARISON[0])
        assert points[0] == pytest.approx((0.0, 0.0))
        assert points[-1] == pytest.approx((1.0, 1.0))

    def test_roc_curve_passes_through_the_operating_point(self) -> None:
        for row in MODALITY_COMPARISON:
            fpr0, tpr0 = operating_point(row)
            points = roc_curve_points(row)
            closest = min(points, key=lambda p: abs(p[0] - fpr0))
            assert closest[0] == pytest.approx(fpr0, abs=1e-9)
            assert closest[1] == pytest.approx(tpr0, abs=1e-3)

    def test_curve_auc_equals_the_reported_roc_auc(self) -> None:
        """Table, confusion matrix and plotted curve cannot disagree."""
        for row in MODALITY_COMPARISON:
            assert model_auc(row) == pytest.approx(row.roc_auc, abs=2e-3)
            assert curve_auc(roc_curve_points(row)) == pytest.approx(
                row.roc_auc, abs=2e-3
            )

    def test_roc_auc_ordering_is_multimodal_first(self) -> None:
        aucs = {c["variant"]: c["auc"] for c in curves_payload()["roc"]}
        assert aucs["multimodal"] > aucs["visual_only"] > aucs["audio_only"]

    def test_pr_curves_are_monotone_in_recall_and_ordered(self) -> None:
        payload = curves_payload()
        for curve in payload["pr"]:
            points = curve["points"]
            assert len(points) > 10
            assert all(a[0] <= b[0] + 1e-9 for a, b in zip(points, points[1:]))
            assert all(0.0 <= p[1] <= 1.0 for p in points)
        aps = {c["variant"]: c["average_precision"] for c in payload["pr"]}
        assert aps["multimodal"] > aps["visual_only"] > aps["audio_only"]


# ---------------------------------------------------------------------------
# Fusion-weight ablation
# ---------------------------------------------------------------------------


class TestWeightAblation:
    def test_sweeps_every_weight_from_audio_only_to_visual_only(self) -> None:
        table = ablation_table(configured_alpha=0.6)
        assert [row["alpha"] for row in table["rows"]] == list(ABLATION_ALPHAS)
        assert table["rows"][0]["alpha"] == 0.0
        assert table["rows"][-1]["alpha"] == 1.0
        for row in table["rows"]:
            assert row["audio_weight"] == pytest.approx(1.0 - row["alpha"])
            assert 0.0 <= row["accuracy"] <= 1.0
            assert 0.0 <= row["roc_auc"] <= 1.0

    def test_configured_weights_are_the_swept_optimum(self) -> None:
        table = ablation_table(configured_alpha=0.6)
        assert table["best_alpha"] == pytest.approx(0.6)
        assert table["configured_alpha"] == pytest.approx(0.6)
        assert table["configured_audio_weight"] == pytest.approx(0.4)
        assert table["configured_matches_best"] is True

    def test_accuracy_peaks_at_the_configured_weight(self) -> None:
        table = ablation_table(configured_alpha=0.6)
        best = max(row["accuracy"] for row in table["rows"])
        at_configured = next(
            row["accuracy"] for row in table["rows"] if row["alpha"] == 0.6
        )
        assert at_configured == pytest.approx(best)

    def test_fusion_beats_both_single_modality_baselines(self) -> None:
        table = ablation_table(configured_alpha=0.6)
        assert table["chosen_vs_audio_only_pp"] > 0
        assert table["chosen_vs_visual_only_pp"] > 0

    def test_evaluation_reuses_the_production_fusion_engine(self) -> None:
        """The ablation scores the real FusionEngine, not a reimplementation."""
        from detection.services.fusion import FusionEngine

        alpha = 0.6
        row = evaluate_alpha(alpha)
        expected_tp = 0
        for sample in VALIDATION_SET:
            outcome = FusionEngine.fuse(
                visual_likelihood=sample.visual,
                audio_likelihood=sample.audio,
                weight_visual=alpha,
                weight_audio=0.4,
                threshold=0.5,
            )
            if sample.is_deepfake and outcome.label == "deepfake":
                expected_tp += 1
        assert row.tp == expected_tp

    def test_validation_set_is_balanced_and_deterministic(self) -> None:
        assert len(VALIDATION_SET) == 200
        assert sum(1 for s in VALIDATION_SET if s.is_deepfake) == 100
        # Rebuilding the table twice yields identical numbers (fixed seed).
        assert ablation_table(0.5, 0.6)["rows"] == ablation_table(0.5, 0.6)["rows"]


# ---------------------------------------------------------------------------
# Rendered figures + API wiring
# ---------------------------------------------------------------------------


class TestFigures:
    @pytest.mark.parametrize(
        "name", ["confusion_matrices", "roc_curve", "pr_curve", "weight_ablation"]
    )
    def test_figure_renders_as_png(self, name: str) -> None:
        clear_cache()
        png = figure_png(name)
        if png is None:
            pytest.skip("matplotlib is not installed in this environment")
        assert png.startswith(PNG_MAGIC)
        assert len(png) > 2000

    def test_figures_payload_is_base64_encoded(self) -> None:
        payload = figures_payload()
        if not payload:
            pytest.skip("matplotlib is not installed in this environment")
        raw = base64.b64decode(payload["roc_curve"]["png_base64"])
        assert raw.startswith(PNG_MAGIC)

    def test_unknown_figure_returns_none(self) -> None:
        assert figure_png("does-not-exist") is None


class TestBenchmarkEndpointExposesAnalytics:
    def test_endpoint_returns_curves_ablation_and_figures(self) -> None:
        payload = APIClient().get("/api/evaluations/benchmark").json()

        # Existing contract is preserved...
        assert payload["modality_comparison"]["ordering_satisfied"] is True
        assert len(payload["cross_dataset"]["table"]) >= 3

        # ...alongside the new evaluation analytics.
        curves = payload["curves"]
        assert len(curves["roc"]) == 3
        assert len(curves["pr"]) == 3
        assert len(curves["confusion_matrices"]) == 3
        assert "FP" in curves["glossary"]

        ablation = payload["weight_ablation"]
        assert len(ablation["rows"]) == 11
        assert ablation["best_alpha"] == pytest.approx(0.6)
        assert ablation["configured_matches_best"] is True

        figures = payload["figures"]
        if figures:  # matplotlib available
            assert "roc_curve" in figures
            assert "pr_curve" in figures
            assert "weight_ablation" in figures
