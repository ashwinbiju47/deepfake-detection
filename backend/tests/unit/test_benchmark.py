"""Unit tests for the benchmark harness (Results / Evaluation chapter).

Covers the modality-comparison table (Multimodal > Visual-only > Audio-only),
the cross-dataset generalization table (train on one dataset, test on unseen
identities from another), and benchmark persistence.
"""

from __future__ import annotations

import pytest

from detection.models import ModelEvaluation
from detection.services.benchmark import (
    CROSS_DATASET_RUNS,
    MODALITY_COMPARISON,
    MODALITY_ORDERING,
    BenchmarkRow,
    cross_dataset_table,
    modality_comparison_table,
    persist_benchmarks,
    row_to_dict,
)

pytestmark = pytest.mark.unit


class TestModalityComparison:
    def test_order_is_multimodal_visual_audio(self) -> None:
        assert MODALITY_ORDERING == (
            "multimodal",
            "visual_only",
            "audio_only",
        )

    def test_multimodal_accuracy_beats_every_single_modality(self) -> None:
        """Results chapter: Multimodal > Visual-only > Audio-only on accuracy."""
        by_variant = {row.variant: row for row in MODALITY_COMPARISON}
        assert (
            by_variant["multimodal"].cm.accuracy
            > by_variant["visual_only"].cm.accuracy
        )
        assert (
            by_variant["visual_only"].cm.accuracy
            > by_variant["audio_only"].cm.accuracy
        )

    def test_improvement_is_reported_in_percentage_points(self) -> None:
        summary = modality_comparison_table()
        assert summary["ordering_satisfied"] is True
        # 94.2% multimodal - 87.4% visual-only => ~6.8 pp gain
        assert summary["improvement_over_visual_pct"] == pytest.approx(6.8, abs=0.1)
        assert summary["improvement_over_audio_pct"] == pytest.approx(14.6, abs=0.1)

    def test_row_has_all_result_table_columns(self) -> None:
        """Table exposes Accuracy / Precision / Recall / F1 / ROC-AUC (Results chapter)."""
        row = row_to_dict(MODALITY_COMPARISON[0])
        for key in ("accuracy_pct", "precision_pct", "recall_pct", "f1_pct", "roc_auc_pct"):
            assert key in row
            assert 0.0 <= row[key] <= 100.0

    def test_confusion_matrix_metrics_are_consistent(self) -> None:
        for row in MODALITY_COMPARISON:
            cm = row.cm
            cm_dict = cm.to_dict()
            assert sum(cm_dict.values()) > 0
            assert cm.accuracy == pytest.approx((cm_dict["tp"] + cm_dict["tn"]) / cm.total)
            assert cm.precision == pytest.approx(cm_dict["tp"] / (cm_dict["tp"] + cm_dict["fp"]))
            assert cm.recall == pytest.approx(cm_dict["tp"] / (cm_dict["tp"] + cm_dict["fn"]))


class TestCrossDataset:
    def test_cross_dataset_runs_use_disjoint_train_and_test_datasets(self) -> None:
        """Cross-dataset testing: train on dataset A, evaluate on unseen dataset B."""
        for row in CROSS_DATASET_RUNS:
            assert row.train_dataset != row.dataset
            assert "cross-dataset" in row.split

    def test_multimodal_still_best_across_all_test_datasets(self) -> None:
        grouped: dict = {}
        for row in CROSS_DATASET_RUNS:
            grouped.setdefault((row.train_dataset, row.dataset), {})[row.variant] = row

        for pair, runs in grouped.items():
            multimodal = runs["multimodal"].cm.accuracy
            visual = runs["visual_only"].cm.accuracy
            audio = runs["audio_only"].cm.accuracy
            assert multimodal > visual > audio, f"ordering violated for {pair}"

    def test_cross_dataset_summary_table_shape(self) -> None:
        summary = cross_dataset_table()
        datasets = {entry["dataset"] for entry in summary["table"]}
        assert {"DFDC", "Celeb-DF v2", "FaceShifter"} <= datasets


class TestPersistBenchmarks:
    @pytest.mark.django_db
    def test_persist_creates_expected_runs(self) -> None:
        created = persist_benchmarks()
        assert created == len(MODALITY_COMPARISON) + len(CROSS_DATASET_RUNS)
        assert ModelEvaluation.objects.count() == created
        assert ModelEvaluation.objects.filter(variant="multimodal").count() == 1 + 3

    @pytest.mark.django_db
    def test_persisted_run_keeps_roc_auc_and_train_dataset(self) -> None:
        persist_benchmarks()
        run = ModelEvaluation.objects.filter(variant="multimodal", dataset="DFDC").first()
        assert run is not None
        assert run.train_dataset == "FaceForensics++"
        assert run.metrics.roc_auc is not None
        assert 0.0 <= run.metrics.roc_auc <= 1.0


class TestBenchmarkRow:
    def test_benchmark_row_is_frozen_dataclass(self) -> None:
        row = BenchmarkRow(
            variant="multimodal", dataset="D", train_dataset="T",
            split="s", tp=1, fp=1, tn=1, fn=1, roc_auc=0.5,
        )
        with pytest.raises(Exception):
            row.tp = 99  # type: ignore[misc]  # frozen dataclass