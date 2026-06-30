"""Property-based test for Upload_Service valid-upload behavior (Task 3.2).

# Feature: deepfake-detection-platform, Property 1: Valid uploads create exactly one session and return its id

**Validates: Requirements 1.1, 1.6**

For any file input whose declared format is a Supported_Format (MP4 or AVI),
whose size is in the inclusive range [1, 52428800] bytes, and which decodes as a
valid video, ``UploadService.submit_file`` SHALL accept it, create exactly one
``AnalysisSession``, and return a non-null session identifier.

Test design notes:
* The decodability probe is dependency-injected with a stub returning ``True``
  (valid decode) so no real video bytes are decoded — the property is about the
  accept/create/return contract, not OpenCV behavior.
* Content bytes are kept small (still within the valid ``[1, 52428800]`` range)
  so 100+ Hypothesis examples run quickly; the injected probe makes the actual
  content irrelevant to the decode decision.
* ``@pytest.mark.django_db`` wraps the whole test function in one transaction, so
  sessions created across Hypothesis examples accumulate until rollback. The
  assertion therefore checks the *delta* (exactly one new session per accepted
  upload) rather than an absolute count.
* ``MEDIA_ROOT`` is redirected to a temp directory so the per-session media the
  service writes on acceptance lands in a throwaway location.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from hypothesis import given
from hypothesis import strategies as st

from detection.models import AnalysisSession
from detection.services.upload import UploadService

# Declared formats that are valid per Supported_Format (Requirement 1.1).
_SUPPORTED_EXTENSIONS = ["mp4", "avi"]

# Keep generated content small but strictly within the valid [1, 52428800] range
# so the suite stays fast across >=100 examples.
_MAX_CONTENT_BYTES = 4096


def _always_decodable(_path: str) -> bool:
    """Decodability probe stub: every staged file decodes as valid video."""
    return True


# Smart generators constrained to the valid input space:
#  * a non-empty stem made of filename-safe characters,
#  * a supported extension, and
#  * content whose length is in [1, _MAX_CONTENT_BYTES] (a subset of the valid
#    [1, 52428800] byte range).
_filename_stem = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=32,
)
_supported_extension = st.sampled_from(_SUPPORTED_EXTENSIONS)
_valid_content = st.binary(min_size=1, max_size=_MAX_CONTENT_BYTES)


@pytest.mark.property
@pytest.mark.django_db
@given(stem=_filename_stem, extension=_supported_extension, content=_valid_content)
def test_valid_uploads_create_exactly_one_session_and_return_its_id(
    settings, tmp_path, stem, extension, content
):
    # Redirect transient media writes to a throwaway directory.
    settings.MEDIA_ROOT = str(tmp_path)

    # A file whose declared format is supported and whose size is within range.
    filename = f"{stem}.{extension}"
    upload = SimpleUploadedFile(
        name=filename, content=content, content_type="video/mp4"
    )
    # Sanity guards on the generated input space.
    assert upload.size is not None and 1 <= upload.size <= settings.MAX_UPLOAD_SIZE_BYTES

    service = UploadService(probe=_always_decodable)

    count_before = AnalysisSession.objects.count()
    result = service.submit_file(upload)
    count_after = AnalysisSession.objects.count()

    # Accepted, with a non-null session id and no error.
    assert result["accepted"] is True
    assert result["error_code"] is None
    session_id = result["session_id"]
    assert session_id is not None

    # Exactly one AnalysisSession was created by this accepted upload.
    assert count_after == count_before + 1

    # The returned id refers to a real, newly created session.
    created = AnalysisSession.objects.get(id=session_id)
    assert str(created.id) == str(session_id)
    assert created.status == AnalysisSession.Status.QUEUED
    assert created.source_type == AnalysisSession.SourceType.FILE
    # The session stores only a filename reference, never media bytes (Req 9.3),
    # and the accepted media was written to the per-session transient dir.
    assert created.source_ref == Path(filename).name
