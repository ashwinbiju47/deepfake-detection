"""Property 21: Non-HTTP(S) URL schemes are rejected with scheme guidance (Task 14.2, Requirement 10.3).

Feature: deepfake-detection-platform, Property 21: Non-HTTP(S) URL schemes are rejected with scheme guidance
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.services.upload import UploadService

pytestmark = [pytest.mark.django_db, pytest.mark.property]


@given(scheme=st.sampled_from(["ftp", "file", "gopher", "rtsp", "sftp", "data"]))
@settings(max_examples=100)
def test_property_non_https_schemes_rejected(scheme: str) -> None:
    """Property 21: Non-HTTP(S) URL schemes are rejected with BAD_SCHEME error."""
    service = UploadService()

    # Enable external URL flag for setting
    with pytest.MonkeyPatch.context() as m:
        m.setattr("django.conf.settings.EXTERNAL_URL_ENABLED", True)
        url = f"{scheme}://example.com/video.mp4"
        res = service.submit_url(url)

        assert res["accepted"] is False
        assert res["error_code"] == "BAD_SCHEME"
        assert "Only HTTP and HTTPS URL schemes are supported" in (res["message"] or "")
