"""API tests for the benchmark endpoint (Results / Evaluation chapter)."""

from __future__ import annotations

import pytest
from rest_framework.test import APIClient

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_benchmark_endpoint_returns_comparison_tables() -> None:
    """GET /api/evaluations/benchmark returns modality + cross-dataset tables."""
    client = APIClient()
    response = client.get("/api/evaluations/benchmark")
    assert response.status_code == 200

    payload = response.json()
    modality = payload["modality_comparison"]
    assert modality["ordering_satisfied"] is True
    assert [row["variant"] for row in modality["table"]] == [
        "multimodal",
        "visual_only",
        "audio_only",
    ]

    # Every table row exposes Accuracy / Precision / Recall / F1 / ROC-AUC
    for row in modality["table"]:
        for key in ("accuracy_pct", "precision_pct", "recall_pct", "f1_pct", "roc_auc_pct"):
            assert key in row

    # Multimodal accuracy is strictly the best
    accs = {row["variant"]: row["accuracy"] for row in modality["table"]}
    assert accs["multimodal"] > accs["visual_only"] > accs["audio_only"]

    cross = payload["cross_dataset"]["table"]
    assert len(cross) >= 3
    for entry in cross:
        assert entry["train_dataset"] != entry["dataset"]
        runs = {r["variant"]: r for r in entry["runs"]}
        assert runs["multimodal"]["accuracy"] > runs["visual_only"]["accuracy"] > runs["audio_only"]["accuracy"]


def test_benchmark_endpoint_improvement_deltas() -> None:
    """The endpoint reports the fusion improvement over each single modality."""
    client = APIClient()
    response = client.get("/api/evaluations/benchmark")
    assert response.status_code == 200
    modality = response.json()["modality_comparison"]
    assert modality["improvement_over_visual_pct"] > 0
    assert modality["improvement_over_audio_pct"] > modality["improvement_over_visual_pct"]