"""Upload_Service: ordered, fail-fast input validation and file submission.

Implements Requirement 1 (Video File Upload). Validation runs **before** any
``AnalysisSession`` is created so a rejected input never leaves a session behind
(Requirements 1.2-1.5). Only when every check passes is exactly one
``AnalysisSession`` (status ``QUEUED``) created and the accepted upload stored in
the transient per-session media directory (Requirements 1.1, 1.6).

Validation pipeline (ordered, fail-fast) — design "Upload_Service" section:

1. **Format check** — file extension against ``SUPPORTED_VIDEO_FORMATS``
   (MP4/AVI). Failure -> ``UNSUPPORTED_FORMAT``.
2. **Size check** — ``1 <= size_bytes <= MAX_UPLOAD_SIZE_BYTES`` (50 MB).
   ``0`` bytes -> ``EMPTY_FILE``; ``> limit`` -> ``TOO_LARGE``.
3. **Decodability probe** — declared-supported but undecodable -> ``UNDECODABLE``.

The decodability probe is **dependency-injected** (``probe`` constructor arg) so
tests can supply a deterministic stub instead of decoding real video bytes. The
default probe uses OpenCV/FFmpeg via :func:`opencv_probe`.

The error codes mirror the design's "Input Validation Errors" table exactly:
``EMPTY_FILE``, ``UNSUPPORTED_FORMAT``, ``TOO_LARGE``, ``UNDECODABLE``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional, TypedDict

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction

from detection.models import AnalysisSession
from detection.storage import session_media_dir


# ---------------------------------------------------------------------------
# Result type and error codes
# ---------------------------------------------------------------------------
class UploadResult(TypedDict):
    """Outcome of an Upload_Service submission (design: Upload_Service)."""

    accepted: bool
    session_id: Optional[str]  # set iff accepted
    error_code: Optional[str]  # one of the ERROR_* codes below; None iff accepted
    message: Optional[str]


# Typed error codes — mirror the design "Input Validation Errors" table.
ERROR_EMPTY_FILE = "EMPTY_FILE"
ERROR_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
ERROR_TOO_LARGE = "TOO_LARGE"
ERROR_UNDECODABLE = "UNDECODABLE"


# Type of the decodability probe: given a path to the stored bytes, return True
# iff the content decodes as a valid video.
DecodeProbe = Callable[[str], bool]


def opencv_probe(path: str) -> bool:
    """Default decodability probe using OpenCV (FFmpeg-backed).

    Returns ``True`` iff the file at ``path`` can be opened as a video and at
    least one frame can be decoded. Any failure to import/open/decode is treated
    as "undecodable" (returns ``False``) rather than raising, so a malformed but
    correctly-extensioned file is reported as ``UNDECODABLE`` (Requirement 1.5).

    OpenCV is imported lazily so the service module can be imported (and the
    pure validation logic unit-tested with a stub probe) in environments where
    the native OpenCV/FFmpeg libraries are unavailable.
    """
    try:
        import cv2  # noqa: PLC0415 (lazy import by design)
    except Exception:  # pragma: no cover - environment without OpenCV
        # Cannot verify decodability; fail closed (treat as undecodable).
        return False

    capture = None
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return False
        ok, _frame = capture.read()
        return bool(ok)
    except Exception:
        return False
    finally:
        if capture is not None:
            capture.release()


class UploadService:
    """Receives, validates, and accepts video inputs (Requirement 1).

    Parameters
    ----------
    probe:
        Decodability probe used by step 3 of the pipeline. Defaults to
        :func:`opencv_probe`. Injectable so tests can supply a deterministic
        stub without real media.
    """

    def __init__(self, probe: Optional[DecodeProbe] = None) -> None:
        self._probe: DecodeProbe = probe or opencv_probe

    # -- public API --------------------------------------------------------
    def submit_file(self, upload: UploadedFile) -> UploadResult:
        """Validate and (on success) accept an uploaded video file.

        Runs the ordered fail-fast pipeline. On the first failing check returns
        a rejection ``UploadResult`` with the matching ``error_code`` and a
        human-readable message, creating **no** ``AnalysisSession``. On success
        creates exactly one ``AnalysisSession`` (status ``QUEUED``), stores the
        bytes in the transient per-session media dir, and returns its id.
        """
        # 1. Format check (extension against the supported set).
        if not self._is_supported_format(upload.name):
            return self._reject(
                ERROR_UNSUPPORTED_FORMAT,
                "Unsupported file format. Supported formats are: "
                f"{self._supported_formats_display()}.",
            )

        # 2. Size check (zero-byte -> empty; oversize -> too large).
        size_bytes = upload.size if upload.size is not None else 0
        if size_bytes <= 0:
            return self._reject(ERROR_EMPTY_FILE, "The uploaded file is empty.")
        if size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
            return self._reject(
                ERROR_TOO_LARGE,
                "The uploaded file exceeds the 50MB "
                f"({settings.MAX_UPLOAD_SIZE_BYTES} bytes) size limit.",
            )

        # 3. Decodability probe — stage the bytes to a temp file and probe it.
        #    The temp file lets us reject undecodable input *before* creating a
        #    session, so no session/media dir is left behind on rejection.
        with tempfile.NamedTemporaryFile(
            suffix=self._suffix(upload.name), delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._write_upload(upload, tmp_path)
            if not self._probe(str(tmp_path)):
                return self._reject(
                    ERROR_UNDECODABLE,
                    "The file could not be read as a valid video.",
                )

            # All checks passed: create exactly one session and store the media.
            session = self._accept(upload, tmp_path)
            return UploadResult(
                accepted=True,
                session_id=str(session.id),
                error_code=None,
                message=None,
            )
        finally:
            # Always remove the staging temp file (the accepted copy lives in
            # the session media dir).
            tmp_path.unlink(missing_ok=True)

    def submit_url(self, url: str) -> UploadResult:  # pragma: no cover
        """External URL intake (Requirement 10).

        Extension point only. Gated behind ``EXTERNAL_URL_ENABLED`` and
        implemented in Task 14 (scheme/reachability/streaming-size-guard). Kept
        here so callers have a stable surface to wire against.
        """
        raise NotImplementedError(
            "URL submission is implemented in Task 14 (EXTERNAL_URL_ENABLED)."
        )

    # -- validation helpers ------------------------------------------------
    def _is_supported_format(self, filename: Optional[str]) -> bool:
        ext = self._extension(filename)
        return ext in self._supported_extensions()

    @staticmethod
    def _supported_extensions() -> set[str]:
        return {fmt.strip().lower().lstrip(".") for fmt in settings.SUPPORTED_VIDEO_FORMATS}

    @staticmethod
    def _supported_formats_display() -> str:
        return ", ".join(
            fmt.strip().upper() for fmt in settings.SUPPORTED_VIDEO_FORMATS
        )

    @staticmethod
    def _extension(filename: Optional[str]) -> str:
        if not filename:
            return ""
        return Path(filename).suffix.lower().lstrip(".")

    @classmethod
    def _suffix(cls, filename: Optional[str]) -> str:
        ext = cls._extension(filename)
        return f".{ext}" if ext else ""

    # -- side-effecting helpers --------------------------------------------
    @staticmethod
    def _write_upload(upload: UploadedFile, destination: Path) -> None:
        """Stream the uploaded file's chunks to ``destination``."""
        # Reset the read pointer in case the file was inspected upstream.
        try:
            upload.seek(0)
        except (AttributeError, ValueError, OSError):
            pass
        with open(destination, "wb") as out:
            for chunk in upload.chunks():
                out.write(chunk)

    @staticmethod
    def _accept(upload: UploadedFile, staged_path: Path) -> AnalysisSession:
        """Create exactly one QUEUED session and store media transiently.

        Wrapped in a transaction so a storage failure does not leave a dangling
        session record.
        """
        with transaction.atomic():
            session = AnalysisSession.objects.create(
                source_type=AnalysisSession.SourceType.FILE,
                source_ref=Path(upload.name).name if upload.name else "upload",
                status=AnalysisSession.Status.QUEUED,
                media_state=AnalysisSession.MediaState.PRESENT,
            )
            media_dir = session_media_dir(session.id)
            media_dir.mkdir(parents=True, exist_ok=True)
            stored_name = Path(upload.name).name if upload.name else "upload"
            shutil.copyfile(staged_path, media_dir / stored_name)
        return session

    @staticmethod
    def _reject(error_code: str, message: str) -> UploadResult:
        return UploadResult(
            accepted=False,
            session_id=None,
            error_code=error_code,
            message=message,
        )
