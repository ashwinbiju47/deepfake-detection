"""Upload_Service: ordered, fail-fast input validation and file submission.

Implements Requirement 1 (media file upload), extended to accept **video,
image, and audio** files. URL intake has been removed: the platform only
accepts direct file uploads.

Validation runs **before** any ``AnalysisSession`` is created so a rejected
input never leaves a session behind (Requirements 1.2-1.5). Only when every
check passes is exactly one ``AnalysisSession`` (status ``QUEUED``) created and
the accepted upload stored in the transient per-session media directory
(Requirements 1.1, 1.6).

Validation pipeline (ordered, fail-fast) — design "Upload_Service" section:

1. **Kind + format check** — the extension is matched against the supported
   sets for video / image / audio (``SUPPORTED_VIDEO_FORMATS``,
   ``SUPPORTED_IMAGE_FORMATS``, ``SUPPORTED_AUDIO_FORMATS``). The matched kind
   is recorded on the session. Failure -> ``UNSUPPORTED_FORMAT``.
2. **Size check** — ``1 <= size_bytes <= MAX_UPLOAD_SIZE_BYTES`` (50 MB).
   ``0`` bytes -> ``EMPTY_FILE``; ``> limit`` -> ``TOO_LARGE``.
3. **Decodability probe** — the probe is chosen by media kind (OpenCV for
   video, OpenCV image decode for image, Librosa/FFmpeg for audio).
   Declared-supported but undecodable -> ``UNDECODABLE``.

The decodability probe is **dependency-injected** (``probe`` constructor arg)
so tests can supply a deterministic stub instead of decoding real media bytes.
The default probes are real and lazy-imported, so this module imports cleanly
without OpenCV/Librosa present.

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
    # Media kind of the accepted upload (echoed for the intake UI).
    media_kind: Optional[str]


# Typed error codes — mirror the design "Input Validation Errors" table.
ERROR_EMPTY_FILE = "EMPTY_FILE"
ERROR_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
ERROR_TOO_LARGE = "TOO_LARGE"
ERROR_UNDECODABLE = "UNDECODABLE"
ERROR_ENQUEUE_FAILED = "ENQUEUE_FAILED"


# Type of the decodability probe: given a path to the stored bytes, return True
# iff the content decodes as valid media of its kind.
DecodeProbe = Callable[[str], bool]
Enqueuer = Callable[[AnalysisSession], bool]

# Mapping extension -> media kind, built from the settings lists.
_VIDEO_EXTS = "mp4 avi mov mkv webm".split()
_IMAGE_EXTS = "jpg jpeg png webp bmp".split()
_AUDIO_EXTS = "wav mp3 flac ogg m4a".split()


def default_enqueuer(session: AnalysisSession) -> bool:
    """Default task enqueuer using Celery delay."""
    try:
        from detection.tasks import analyze_session

        analyze_session.delay(str(session.id))
        return True
    except Exception:
        # Update session status to FAILED so it never remains in-progress
        session.status = AnalysisSession.Status.FAILED
        session.save(update_fields=["status"])
        return False


def _env_extensions(setting_name: str, fallback: list[str]) -> list[str]:
    """Read an extension list from settings (falling back if the attr is gone)."""
    raw = getattr(settings, setting_name, None)
    if not raw:
        return fallback
    return [str(fmt).strip().lower().lstrip(".") for fmt in raw if str(fmt).strip()]


def media_kind_for(filename: Optional[str]) -> Optional[str]:
    """Classify a filename into a media kind, or ``None`` if unsupported.

    Pure function of the extension so tests can exercise it without any media
    library. Returns one of ``AnalysisSession.MediaKind`` values.
    """
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if not ext:
        return None
    if ext in _env_extensions("SUPPORTED_VIDEO_FORMATS", _VIDEO_EXTS):
        return AnalysisSession.MediaKind.VIDEO
    if ext in _env_extensions("SUPPORTED_IMAGE_FORMATS", _IMAGE_EXTS):
        return AnalysisSession.MediaKind.IMAGE
    if ext in _env_extensions("SUPPORTED_AUDIO_FORMATS", _AUDIO_EXTS):
        return AnalysisSession.MediaKind.AUDIO
    return None


def _video_probe(path: str) -> bool:
    """Probe decodability of a video with OpenCV (lazy import)."""
    try:
        import cv2  # noqa: PLC0415 (lazy import by design)
    except Exception:  # pragma: no cover - environment without OpenCV
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


def _image_probe(path: str) -> bool:
    """Probe decodability of an image with OpenCV (lazy import)."""
    try:
        import cv2  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return False
    try:
        data = open(path, "rb").read()
        image = cv2.imdecode(
            __import__("numpy").frombuffer(data, dtype="uint8"),
            cv2.IMREAD_COLOR,
        )
        return image is not None and getattr(image, "size", 0) > 0
    except Exception:
        return False


def _audio_probe(path: str) -> bool:
    """Probe decodability of an audio file with Librosa/FFmpeg (lazy import)."""
    try:
        import librosa  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return False
    try:
        y, _sr = librosa.load(path, sr=None, mono=True, duration=1.0)
        return y is not None and len(y) > 0
    except Exception:
        return False


_KIND_PROBES: dict[str, Callable[[str], bool]] = {
    AnalysisSession.MediaKind.VIDEO: _video_probe,
    AnalysisSession.MediaKind.IMAGE: _image_probe,
    AnalysisSession.MediaKind.AUDIO: _audio_probe,
}


class UploadService:
    """Receives, validates, and accepts video/image/audio inputs (Requirement 1).

    Parameters
    ----------
    probe:
        Decodability probe used by step 3 of the pipeline. Defaults to a
        per-kind real probe (``_KIND_PROBES``). Injectable so tests can supply
        a deterministic stub without real media.
    enqueuer:
        Task enqueuer used after acceptance. Defaults to Celery delay.
    """

    def __init__(
        self,
        probe: Optional[DecodeProbe] = None,
        enqueuer: Optional[Enqueuer] = None,
    ) -> None:
        self._probe: Optional[DecodeProbe] = probe
        self._enqueuer: Enqueuer = enqueuer or default_enqueuer

    def _probe_for(self, media_kind: str) -> DecodeProbe:
        """Return the injected probe, or the real probe for ``media_kind``."""
        if self._probe is not None:
            return self._probe
        return _KIND_PROBES.get(
            media_kind,
            _video_probe,  # unknown kinds never reach here (kind check is first)
        )

    def _enqueue_analysis(self, session: AnalysisSession) -> bool:
        return self._enqueuer(session)

    # -- public API --------------------------------------------------------
    def submit_file(self, upload: UploadedFile) -> UploadResult:
        """Validate and (on success) accept an uploaded media file.

        Runs the ordered fail-fast pipeline. On the first failing check returns
        a rejection ``UploadResult`` with the matching ``error_code`` and a
        human-readable message, creating **no** ``AnalysisSession``. On success
        creates exactly one ``AnalysisSession`` (status ``QUEUED``), records the
        detected ``media_kind``, stores the bytes in the transient per-session
        media dir, and returns its id.
        """
        # 1. Kind + format check (extension against the supported sets).
        media_kind = media_kind_for(upload.name)
        if media_kind is None:
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
            if not self._probe_for(media_kind)(str(tmp_path)):
                return self._reject(
                    ERROR_UNDECODABLE,
                    "The file could not be read as a valid "
                    f"{self._kind_display(media_kind)}.",
                )

            # All checks passed: create session, store media, and enqueue task.
            session = self._accept(upload, tmp_path, media_kind)

            # Enqueue analysis task (Requirement 6.1, 6.2)
            enqueue_success = self._enqueue_analysis(session)
            if not enqueue_success:
                return self._reject(
                    ERROR_ENQUEUE_FAILED,
                    "Failed to enqueue analysis job for processing.",
                )

            return UploadResult(
                accepted=True,
                session_id=str(session.id),
                error_code=None,
                message=None,
                media_kind=media_kind,
            )
        finally:
            # Always remove the staging temp file (the accepted copy lives in
            # the session media dir).
            tmp_path.unlink(missing_ok=True)

    # -- validation helpers ------------------------------------------------
    def _supported_formats_display(self) -> str:
        video = ", ".join(
            fmt.strip().upper() for fmt in _env_extensions("SUPPORTED_VIDEO_FORMATS", _VIDEO_EXTS)
        )
        image = ", ".join(
            fmt.strip().upper() for fmt in _env_extensions("SUPPORTED_IMAGE_FORMATS", _IMAGE_EXTS)
        )
        audio = ", ".join(
            fmt.strip().upper() for fmt in _env_extensions("SUPPORTED_AUDIO_FORMATS", _AUDIO_EXTS)
        )
        return f"videos ({video}), images ({image}), audio ({audio})"

    @staticmethod
    def _kind_display(media_kind: str) -> str:
        return {
            AnalysisSession.MediaKind.VIDEO: "video",
            AnalysisSession.MediaKind.IMAGE: "image",
            AnalysisSession.MediaKind.AUDIO: "audio",
        }.get(media_kind, media_kind)

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
    def _accept(upload: UploadedFile, staged_path: Path, media_kind: str) -> AnalysisSession:
        """Create exactly one QUEUED session and store media transiently.

        Wrapped in a transaction so a storage failure does not leave a dangling
        session record.
        """
        with transaction.atomic():
            session = AnalysisSession.objects.create(
                source_type=AnalysisSession.SourceType.FILE,
                source_ref=Path(upload.name).name if upload.name else "upload",
                media_kind=media_kind,
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
            media_kind=None,
        )
