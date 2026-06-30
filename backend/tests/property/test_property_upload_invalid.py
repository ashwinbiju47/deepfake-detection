"""Property-based test for Upload_Service rejection of invalid uploads.

# Feature: deepfake-detection-platform, Property 2: Invalid uploads are rejected with no session and a correct message

Validates: Requirements 1.2, 1.3, 1.4, 1.5

For any file input that is invalid for *exactly one* of the reasons

    {unsupported format, size > 50MB, size == 0 bytes,
     declared-supported-but-undecodable}

``UploadService.submit_file`` must:

  * reject it (``accepted is False``),
  * create **zero** ``AnalysisSession`` records, and
  * return the matching ``error_code`` together with a human-readable message
    that identifies that specific reason.

The decodability probe is dependency-injected, so the "declared-supported but
undecodable" category is produced with a stub probe returning ``False`` while
the other three categories are made invalid purely by filename/size and never
reach the probe (the validation pipeline is ordered and fail-fast).

A lightweight in-memory ``_FakeUpload`` stands in for ``UploadedFile`` so the
oversize category can declare ``size > 52428800`` without allocating 50 MB of
real bytes (the service reads ``upload.size`` directly and only streams
``chunks()`` for the undecodable category, which reaches the probe).
"""

from __future__ import annotations

from typing import Iterator, Optional

import pytest
from django.conf import settings
from hypothesis import given
from hypothesis import strategies as st

from detection.models import AnalysisSession
from detection.services.upload import (
    ERROR_EMPTY_FILE,
    ERROR_TOO_LARGE,
    ERROR_UNDECODABLE,
    ERROR_UNSUPPORTED_FORMAT,
    UploadService,
)

# Invalid categories and the error_code each must yield.
CATEGORY_UNSUPPORTED = "unsupported_format"
CATEGORY_OVERSIZE = "oversize"
CATEGORY_EMPTY = "empty"
CATEGORY_UNDECODABLE = "undecodable"

EXPECTED_CODE = {
    CATEGORY_UNSUPPORTED: ERROR_UNSUPPORTED_FORMAT,
    CATEGORY_OVERSIZE: ERROR_TOO_LARGE,
    CATEGORY_EMPTY: ERROR_EMPTY_FILE,
    CATEGORY_UNDECODABLE: ERROR_UNDECODABLE,
}

# A distinctive fragment of the message for each reason (design: Upload_Service
# "Input Validation Errors"), used to assert the message names the right reason.
EXPECTED_MESSAGE_FRAGMENT = {
    CATEGORY_UNSUPPORTED: "format",
    CATEGORY_OVERSIZE: "50MB",
    CATEGORY_EMPTY: "empty",
    CATEGORY_UNDECODABLE: "valid video",
}


class _FakeUpload:
    """Minimal stand-in for ``django.core.files.uploadedfile.UploadedFile``.

    Exposes only the surface ``UploadService.submit_file`` touches: ``name``,
    ``size``, ``seek`` and ``chunks``. ``size`` is an independent attribute so
    an oversize file can be *declared* without holding 50 MB in memory.
    """

    def __init__(self, name: Optional[str], size: int, content: bytes = b"") -> None:
        self.name = name
        self.size = size
        self._content = content

    def seek(self, *_args, **_kwargs) -> int:
        return 0

    def chunks(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def _supported_extensions() -> set[str]:
    return {fmt.strip().lower().lstrip(".") for fmt in settings.SUPPORTED_VIDEO_FORMATS}


# A pool of extensions that are clearly NOT supported video containers.
_UNSUPPORTED_EXT_POOL = [
    "mkv", "mov", "flv", "wmv", "webm", "txt", "png", "jpg",
    "exe", "pdf", "zip", "gif", "mp3", "wav", "doc", "",
]


@st.composite
def _base_names(draw: st.DrawFn) -> str:
    """A safe, non-empty base filename (no dot, no path separators)."""
    name = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="_-",
            ),
            min_size=1,
            max_size=20,
        )
    )
    return name


@st.composite
def _invalid_uploads(draw: st.DrawFn):
    """Generate (category, _FakeUpload) invalid for exactly one reason."""
    supported = _supported_extensions()
    a_supported_ext = sorted(supported)[0] if supported else "mp4"
    max_bytes = settings.MAX_UPLOAD_SIZE_BYTES

    category = draw(
        st.sampled_from(
            [
                CATEGORY_UNSUPPORTED,
                CATEGORY_OVERSIZE,
                CATEGORY_EMPTY,
                CATEGORY_UNDECODABLE,
            ]
        )
    )
    base = draw(_base_names())

    if category == CATEGORY_UNSUPPORTED:
        # Unsupported extension; size is otherwise valid so the *only* defect
        # is the format. (Rejection happens before the size/probe stages.)
        ext = draw(
            st.sampled_from(_UNSUPPORTED_EXT_POOL).filter(
                lambda e: e.lower() not in supported
            )
        )
        size = draw(st.integers(min_value=1, max_value=max_bytes))
        name = f"{base}.{ext}" if ext else base
        return category, _FakeUpload(name=name, size=size)

    if category == CATEGORY_OVERSIZE:
        # Supported format, but size strictly above the 50 MB limit.
        size = draw(
            st.integers(min_value=max_bytes + 1, max_value=max_bytes * 4)
        )
        return category, _FakeUpload(name=f"{base}.{a_supported_ext}", size=size)

    if category == CATEGORY_EMPTY:
        # Supported format, zero bytes.
        return category, _FakeUpload(name=f"{base}.{a_supported_ext}", size=0)

    # CATEGORY_UNDECODABLE: supported format, valid size, but the probe will
    # report it cannot be decoded. A few real bytes are streamed to the probe.
    size = draw(st.integers(min_value=1, max_value=max_bytes))
    content = draw(st.binary(min_size=1, max_size=64))
    return category, _FakeUpload(
        name=f"{base}.{a_supported_ext}", size=size, content=content
    )


def _service_for(category: str) -> UploadService:
    """Build a service whose injected probe matches the category.

    Only the undecodable category should fail the decodability probe; every
    other category must be invalid for its own reason and would otherwise pass
    the probe (which it never reaches, given fail-fast ordering).
    """
    if category == CATEGORY_UNDECODABLE:
        return UploadService(probe=lambda _path: False)
    return UploadService(probe=lambda _path: True)


@pytest.mark.property
@pytest.mark.django_db
@given(case=_invalid_uploads())
def test_invalid_uploads_rejected_with_no_session_and_correct_message(case) -> None:
    category, upload = case
    service = _service_for(category)

    result = service.submit_file(upload)

    # Rejected, with no accepted session id.
    assert result["accepted"] is False
    assert result["session_id"] is None

    # Zero AnalysisSession records created (validation precedes session creation).
    assert AnalysisSession.objects.count() == 0

    # The matching error_code for this specific reason.
    assert result["error_code"] == EXPECTED_CODE[category]

    # The message identifies that specific reason.
    message = result["message"] or ""
    assert EXPECTED_MESSAGE_FRAGMENT[category].lower() in message.lower()
