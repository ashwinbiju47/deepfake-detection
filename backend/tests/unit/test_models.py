"""Unit tests for ORM model constraints and state enums (Task 2.2).

These example-based tests lock down the persistence-layer invariants that the
design's ERD (design.md, Data Models) and Requirement 9.3 depend on:

* **State/status enum values** mirror the ERD strings exactly. Downstream
  components (Fusion_Engine, Result_Streamer, Media_Purger, Report_Generator)
  and the WebSocket/JSON protocol compare against these literal strings, so a
  drift here would silently break wiring.
* **Nullability of likelihood / score fields.** ``None`` represents the
  no-signal (Req 2.4, 3.4) and inconclusive (Req 4.4) states precisely rather
  than overloading ``0.0``.
* **Media vs. metadata separation (Requirement 9.3).** ``AnalysisSession``
  stores no media bytes: ``source_ref`` is a plain filename/URL string and the
  model exposes no binary/file media field. Actual media lives in the transient
  store keyed by ``session_id`` and is removed by the Media_Purger.

Most checks are pure schema/enum introspection and need no database. The
media-separation persistence check is additionally exercised against a real DB
under ``@pytest.mark.django_db`` to confirm a session round-trips storing only a
string reference.
"""

from __future__ import annotations

import pytest
from django.db import models

from detection.models import (
    AnalysisSession,
    AudioResult,
    FusionResult,
    PurgeRecord,
    Report,
    StreamEvent,
    VisualResult,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# State / status enum values match the design ERD strings exactly.
# ---------------------------------------------------------------------------
class TestEnumValuesMatchERD:
    """Each TextChoices value must equal the literal ERD string."""

    def test_analysis_session_source_type_values(self) -> None:
        assert AnalysisSession.SourceType.FILE == "FILE"
        assert AnalysisSession.SourceType.URL == "URL"
        assert set(AnalysisSession.SourceType.values) == {"FILE", "URL"}

    def test_analysis_session_status_values(self) -> None:
        assert set(AnalysisSession.Status.values) == {
            "QUEUED",
            "PROCESSING",
            "COMPLETED",
            "FAILED",
            "CANCELED",
            "INCONCLUSIVE",
        }

    def test_analysis_session_media_state_values(self) -> None:
        assert set(AnalysisSession.MediaState.values) == {"PRESENT", "PURGED"}

    def test_visual_result_state_values(self) -> None:
        assert set(VisualResult.State.values) == {
            "OK",
            "NO_VISUAL_SIGNAL",
            "VISUAL_ERROR",
        }

    def test_audio_result_state_values(self) -> None:
        assert set(AudioResult.State.values) == {
            "OK",
            "NO_AUDIO_SIGNAL",
            "AUDIO_ERROR",
        }

    def test_fusion_result_label_values(self) -> None:
        # Labels are the lowercase strings used in the WebSocket/JSON protocol.
        assert set(FusionResult.Label.values) == {"authentic", "deepfake"}

    def test_stream_event_type_values(self) -> None:
        assert set(StreamEvent.EventType.values) == {
            "progress",
            "heatmap",
            "result",
            "error",
        }

    def test_purge_record_status_values(self) -> None:
        assert set(PurgeRecord.Status.values) == {"SUCCESS", "FAILED"}

    def test_report_status_values(self) -> None:
        assert set(Report.Status.values) == {"AVAILABLE", "FAILED"}


# ---------------------------------------------------------------------------
# Default state/status values match the ERD defaults.
# ---------------------------------------------------------------------------
class TestEnumDefaults:
    def test_session_defaults_to_queued_and_present(self) -> None:
        assert AnalysisSession._meta.get_field("status").default == "QUEUED"
        assert AnalysisSession._meta.get_field("media_state").default == "PRESENT"

    def test_modality_results_default_to_ok(self) -> None:
        assert VisualResult._meta.get_field("state").default == "OK"
        assert AudioResult._meta.get_field("state").default == "OK"


# ---------------------------------------------------------------------------
# Nullability of likelihood / score fields (no-signal & inconclusive states).
# ---------------------------------------------------------------------------
class TestLikelihoodAndScoreNullability:
    def test_visual_aggregate_likelihood_is_nullable(self) -> None:
        field = VisualResult._meta.get_field("aggregate_likelihood")
        assert isinstance(field, models.FloatField)
        assert field.null is True

    def test_audio_likelihood_is_nullable(self) -> None:
        field = AudioResult._meta.get_field("likelihood")
        assert isinstance(field, models.FloatField)
        assert field.null is True

    def test_fusion_score_is_nullable(self) -> None:
        field = FusionResult._meta.get_field("score")
        assert isinstance(field, models.FloatField)
        assert field.null is True

    def test_fusion_label_is_nullable(self) -> None:
        # No label when inconclusive (Req 4.4).
        field = FusionResult._meta.get_field("label")
        assert field.null is True

    def test_threshold_used_is_not_nullable(self) -> None:
        # The threshold applied is always recorded for a fusion result.
        assert FusionResult._meta.get_field("threshold_used").null is False


# ---------------------------------------------------------------------------
# Media vs. metadata separation (Requirement 9.3): the session stores no media.
# ---------------------------------------------------------------------------
class TestSessionStoresNoMediaBytes:
    def test_source_ref_is_a_string_field(self) -> None:
        field = AnalysisSession._meta.get_field("source_ref")
        assert isinstance(field, models.CharField)
        # A filename/URL fits comfortably in a CharField; it never holds bytes.
        assert field.max_length and field.max_length <= 4096

    def test_session_has_no_media_byte_or_file_fields(self) -> None:
        """AnalysisSession must expose no binary/file/media-bearing field."""
        media_field_types = (
            models.BinaryField,
            models.FileField,  # also covers ImageField (a FileField subclass)
        )
        offending = [
            f.name
            for f in AnalysisSession._meta.get_fields()
            if isinstance(f, media_field_types)
        ]
        assert offending == [], f"unexpected media field(s) on session: {offending}"

    def test_session_field_names_contain_only_metadata(self) -> None:
        """The session's own (non-relation) columns are metadata only."""
        own_fields = {
            f.name
            for f in AnalysisSession._meta.get_fields()
            if isinstance(f, models.Field) and not f.is_relation
        }
        assert own_fields == {
            "id",
            "source_type",
            "source_ref",
            "status",
            "media_state",
            "created_at",
            "started_at",
            "completed_at",
        }


@pytest.mark.django_db
class TestSessionPersistenceStoresOnlyMetadata:
    """End-to-end persistence check: a session round-trips storing only a string ref."""

    def test_session_persists_filename_reference_only(self) -> None:
        session = AnalysisSession.objects.create(
            source_type=AnalysisSession.SourceType.FILE,
            source_ref="clip.mp4",
        )
        reloaded = AnalysisSession.objects.get(pk=session.pk)
        assert reloaded.source_ref == "clip.mp4"
        assert isinstance(reloaded.source_ref, str)
        assert reloaded.status == AnalysisSession.Status.QUEUED
        assert reloaded.media_state == AnalysisSession.MediaState.PRESENT

    def test_modality_likelihoods_and_score_persist_as_null(self) -> None:
        session = AnalysisSession.objects.create(
            source_type=AnalysisSession.SourceType.URL,
            source_ref="https://example.com/clip.mp4",
        )
        visual = VisualResult.objects.create(
            session=session, state=VisualResult.State.NO_VISUAL_SIGNAL
        )
        audio = AudioResult.objects.create(
            session=session, state=AudioResult.State.NO_AUDIO_SIGNAL
        )
        fusion = FusionResult.objects.create(
            session=session, inconclusive=True, threshold_used=0.5
        )

        assert VisualResult.objects.get(pk=visual.pk).aggregate_likelihood is None
        assert AudioResult.objects.get(pk=audio.pk).likelihood is None
        reloaded_fusion = FusionResult.objects.get(pk=fusion.pk)
        assert reloaded_fusion.score is None
        assert reloaded_fusion.label is None
        assert reloaded_fusion.inconclusive is True
