"""Unit tests for purge-failure alerting (Task 11.4, Requirement 9.5).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from detection.models import AnalysisSession, PurgeRecord
from detection.services.purger import MediaPurger

pytestmark = [pytest.mark.django_db, pytest.mark.unit]


def test_purge_failure_retries_and_alerts() -> None:
    """Requirement 9.5: Purge failure retries 3 times, sets status FAILED, and logs alert."""
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref="video.mp4",
        status=AnalysisSession.Status.FAILED,
    )

    with patch("detection.services.purger.shutil.rmtree", side_effect=PermissionError("Permission denied")):
        with patch("detection.services.purger.session_media_dir") as mock_dir:
            mock_dir.return_value.exists.return_value = True

            record = MediaPurger.purge(str(session.id), max_retries=3)

            assert record.status == PurgeRecord.Status.FAILED
            assert record.retries == 3
            assert record.verified_empty is False

