"""Core ORM models for the deepfake-detection platform (Task 2.1).

These models implement the Entity Relationship Diagram in design.md
(Data Models section). Key design invariants encoded here:

* **UUID primary keys** wherever the ERD shows ``uuid`` / ``run_id``.
* **Media vs. metadata separation (Requirement 9.3):** ``AnalysisSession``
  stores no media bytes. ``source_ref`` holds only a filename or URL string;
  the actual media lives in the transient store keyed by ``session_id`` and is
  removed by the ``Media_Purger``. ``FrameHeatmap.overlay_png`` is a *derived*
  visualization (model-activation overlay), explicitly a non-source-media
  report artifact, not the original video.
* **Nullable likelihood / score fields (Requirements 4.4, 2.4, 3.4):** the
  per-modality likelihoods and the fused ``score`` are nullable so the
  no-signal and inconclusive states are represented precisely rather than
  overloaded onto ``0.0``.

Enum/choice values mirror the ERD strings exactly.
"""

from __future__ import annotations

import uuid

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

# Shared validators for likelihood / score values constrained to [0.0, 1.0].
_UNIT_INTERVAL = (MinValueValidator(0.0), MaxValueValidator(1.0))


class AnalysisSession(models.Model):
    """A single end-to-end processing job for one video input (ERD: ANALYSIS_SESSION).

    Stores only non-media metadata. ``source_ref`` is a filename or URL string;
    no media bytes are ever persisted here (Requirement 9.3).
    """

    class SourceType(models.TextChoices):
        FILE = "FILE", "File"
        URL = "URL", "URL"

    class Status(models.TextChoices):
        QUEUED = "QUEUED", "Queued"
        PROCESSING = "PROCESSING", "Processing"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"
        CANCELED = "CANCELED", "Canceled"
        INCONCLUSIVE = "INCONCLUSIVE", "Inconclusive"

    class MediaState(models.TextChoices):
        PRESENT = "PRESENT", "Present"
        PURGED = "PURGED", "Purged"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_type = models.CharField(max_length=8, choices=SourceType.choices)
    # Filename or URL only — never media bytes (Requirement 9.3).
    source_ref = models.CharField(max_length=2048)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.QUEUED
    )
    media_state = models.CharField(
        max_length=8, choices=MediaState.choices, default=MediaState.PRESENT
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"AnalysisSession({self.id}, {self.status})"


class VisualResult(models.Model):
    """Aggregate visual-analysis outcome for a session (ERD: VISUAL_RESULT)."""

    class State(models.TextChoices):
        OK = "OK", "Ok"
        NO_VISUAL_SIGNAL = "NO_VISUAL_SIGNAL", "No visual signal"
        VISUAL_ERROR = "VISUAL_ERROR", "Visual error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="visual_result",
    )
    # Nullable: None represents no/failed visual signal (Requirement 2.4).
    aggregate_likelihood = models.FloatField(
        null=True, blank=True, validators=list(_UNIT_INTERVAL)
    )
    state = models.CharField(max_length=20, choices=State.choices, default=State.OK)
    frames_analyzed = models.IntegerField(default=0)
    faces_isolated = models.IntegerField(default=0)
    error_detail = models.CharField(max_length=1024, blank=True, default="")

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"VisualResult({self.session_id}, {self.state})"


class AudioResult(models.Model):
    """Audio-analysis outcome for a session (ERD: AUDIO_RESULT)."""

    class State(models.TextChoices):
        OK = "OK", "Ok"
        NO_AUDIO_SIGNAL = "NO_AUDIO_SIGNAL", "No audio signal"
        AUDIO_ERROR = "AUDIO_ERROR", "Audio error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="audio_result",
    )
    # Nullable: None represents no/failed audio signal (Requirement 3.4).
    likelihood = models.FloatField(
        null=True, blank=True, validators=list(_UNIT_INTERVAL)
    )
    state = models.CharField(max_length=20, choices=State.choices, default=State.OK)
    error_detail = models.CharField(max_length=1024, blank=True, default="")

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"AudioResult({self.session_id}, {self.state})"


class FusionResult(models.Model):
    """Fused multi-modal classification for a session (ERD: FUSION_RESULT)."""

    class Label(models.TextChoices):
        AUTHENTIC = "authentic", "Authentic"
        DEEPFAKE = "deepfake", "Deepfake"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="fusion_result",
    )
    # Nullable: None iff inconclusive — no modality produced a value (Req 4.4).
    score = models.FloatField(null=True, blank=True, validators=list(_UNIT_INTERVAL))
    # Nullable: no label when inconclusive.
    label = models.CharField(
        max_length=16, choices=Label.choices, null=True, blank=True
    )
    # CSV of the modalities that contributed (subset of {"visual", "audio"}).
    modalities_used = models.CharField(max_length=64, blank=True, default="")
    inconclusive = models.BooleanField(default=False)
    threshold_used = models.FloatField(validators=list(_UNIT_INTERVAL))

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"FusionResult({self.session_id}, score={self.score})"


class FrameHeatmap(models.Model):
    """Grad-CAM XAI artifacts for one analyzed frame (ERD: FRAME_HEATMAP).

    For every analyzed frame we persist the **full XAI triple**:

    * ``original_png`` — the original frame/face crop (ground truth pixels);
    * ``heatmap_png`` — the raw Grad-CAM activation heatmap (jet colormap);
    * ``overlay_png`` — the heatmap alpha-blended over the original frame.

    These are derived visualizations (model-activation overlays), not source
    media — they may persist for reports without violating the GDPR media
    purge (design Key Model Notes).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="frame_heatmaps",
    )
    frame_id = models.CharField(max_length=64)
    original_png = models.BinaryField(null=True, blank=True)
    heatmap_png = models.BinaryField(null=True, blank=True)
    overlay_png = models.BinaryField()
    delivered = models.BooleanField(default=False)

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"FrameHeatmap({self.session_id}, {self.frame_id})"


class StreamEvent(models.Model):
    """Append-only, sequence-numbered stream event log (ERD: STREAM_EVENT).

    Enables resume-without-loss on reconnect (Requirement 5.5).
    """

    class EventType(models.TextChoices):
        PROGRESS = "progress", "Progress"
        HEATMAP = "heatmap", "Heatmap"
        RESULT = "result", "Result"
        ERROR = "error", "Error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="stream_events",
    )
    seq = models.IntegerField()
    type = models.CharField(max_length=16, choices=EventType.choices)
    payload = models.JSONField()
    ts = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["seq"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "seq"], name="unique_session_seq"
            )
        ]

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"StreamEvent({self.session_id}, seq={self.seq}, {self.type})"


class PurgeRecord(models.Model):
    """Media-purge outcome for a session (ERD: PURGE_RECORD)."""

    class Status(models.TextChoices):
        SUCCESS = "SUCCESS", "Success"
        FAILED = "FAILED", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="purge_record",
    )
    status = models.CharField(max_length=8, choices=Status.choices)
    retries = models.IntegerField(default=0)
    verified_empty = models.BooleanField(default=False)
    purged_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"PurgeRecord({self.session_id}, {self.status})"


class Report(models.Model):
    """Downloadable PDF report record for a session (ERD: REPORT)."""

    class Status(models.TextChoices):
        AVAILABLE = "AVAILABLE", "Available"
        FAILED = "FAILED", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="report",
    )
    status = models.CharField(max_length=16, choices=Status.choices)
    generated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"Report({self.session_id}, {self.status})"


class ModelEvaluation(models.Model):
    """A model-evaluation run against a benchmark dataset (ERD: MODEL_EVALUATION).

    ``variant`` identifies which pipeline configuration produced the run:
    ``multimodal`` (fused), ``visual_only``, or ``audio_only``. Recording the
    variant per run lets the platform demonstrate the modality ordering
    Multimodal > Visual-only > Audio-only (Requirement 8 / Results chapter).

    ``train_dataset`` records where the model was trained so cross-dataset
    runs (trained on dataset A, evaluated on dataset B with disjoint
    identities) are auditable; it defaults to the evaluation dataset for
    in-domain runs.
    """

    class Variant(models.TextChoices):
        MULTIMODAL = "multimodal", "Multimodal"
        VISUAL_ONLY = "visual_only", "Visual-only"
        AUDIO_ONLY = "audio_only", "Audio-only"

    run_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.CharField(max_length=128, default="FaceForensics++")
    train_dataset = models.CharField(max_length=128, default="FaceForensics++")
    split = models.CharField(max_length=128, default="held-out test")
    variant = models.CharField(
        max_length=16, choices=Variant.choices, default=Variant.MULTIMODAL
    )
    accuracy = models.FloatField(validators=list(_UNIT_INTERVAL))
    # meets_baseline iff accuracy >= 0.85 (Requirement 8.1).
    meets_baseline = models.BooleanField(default=False)
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"ModelEvaluation({self.run_id}, {self.variant}, acc={self.accuracy})"


class EvaluationMetrics(models.Model):
    """Detailed metrics for an evaluation run (ERD: EVALUATION_METRICS).

    Exactly one metrics object per ``ModelEvaluation`` (1:1 relationship).
    ``roc_auc`` is nullable: it is only recorded when score-level predictions
    (not just the confusion matrix) are available.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.OneToOneField(
        ModelEvaluation,
        on_delete=models.CASCADE,
        related_name="metrics",
    )
    confusion_matrix = models.JSONField()
    precision = models.FloatField(validators=list(_UNIT_INTERVAL))
    recall = models.FloatField(validators=list(_UNIT_INTERVAL))
    f1_score = models.FloatField(validators=list(_UNIT_INTERVAL))
    roc_auc = models.FloatField(
        null=True, blank=True, validators=list(_UNIT_INTERVAL)
    )

    def __str__(self) -> str:  # pragma: no cover - trivial repr
        return f"EvaluationMetrics({self.run_id})"
