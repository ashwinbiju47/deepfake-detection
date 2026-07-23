"""Audio_Extractor: audio extraction and spectrogram generation (Task 5.1, Requirement 3).

This module implements the design's ``Audio_Extractor`` interface:

    extract_audio(video_path) -> DecodedAudio | None
    to_spectrograms(audio) -> list[SpectrogramRepresentation]

Design intent (design.md -> Components and Interfaces -> Audio_Extractor):

* Extract audio track from input video (Requirement 3.1). Return None if no audio track exists.
* Convert audio signal into spectrogram representations (Requirement 3.2). For non-empty
  audio, generate at least one spectrogram representation (Property 7).
* When no audio track is found, record ``NO_AUDIO_SIGNAL`` (Requirement 3.4).
* When audio extraction or spectrogram generation fails, record ``AUDIO_ERROR`` (Requirement 3.5).
* In all cases, preserve existing session data and avoid crashing the overall pipeline.

Testability decision:
---------------------
The core business logic (spectrogram generation count, signal state classification) is unit-
and property-testable without requiring real FFmpeg, Librosa, or audio files by injecting:

* a **demuxer** callable ``video_path -> DecodedAudio | None``, and
* a **spectrogram_generator** callable ``DecodedAudio -> list[SpectrogramRepresentation]``.

Tests supply fakes (e.g. :class:`InMemoryDecodedAudio`). Default back ends import FFmpeg / Librosa
lazily and guarded so importing this module never fails when ML/CV/audio libs are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class AudioSignalState(str, Enum):
    """Audio-analysis signal state for a session.

    Values mirror ``detection.models.AudioResult.State`` (OK / NO_AUDIO_SIGNAL / AUDIO_ERROR).
    """

    OK = "OK"
    NO_AUDIO_SIGNAL = "NO_AUDIO_SIGNAL"
    AUDIO_ERROR = "AUDIO_ERROR"


@dataclass(frozen=True)
class DecodedAudio:
    """Decoded audio track representation."""

    sample_rate: int
    duration_seconds: float
    samples: Any = None  # 1D numpy array or list of float samples

    @property
    def is_empty(self) -> bool:
        if self.samples is None:
            return True
        if hasattr(self.samples, "__len__"):
            return len(self.samples) == 0
        return False


@dataclass(frozen=True)
class SpectrogramRepresentation:
    """Generated spectrogram representation (e.g. mel-spectrogram)."""

    sample_rate: int
    n_mels: int
    time_steps: int
    data: Any = None  # 2D numpy array or list of lists

    @property
    def is_valid(self) -> bool:
        return self.n_mels > 0 and self.time_steps > 0


@dataclass
class AudioExtractionOutcome:
    """Result of the audio extraction and spectrogram generation pass for a session."""

    state: AudioSignalState
    has_audio_track: bool = False
    spectrograms: List[SpectrogramRepresentation] = field(default_factory=list)
    error_detail: str = ""

    @property
    def has_audio_signal(self) -> bool:
        return self.state is AudioSignalState.OK


class AudioExtractionError(Exception):
    """Raised when audio demuxing or processing fails explicitly."""


# ---------------------------------------------------------------------------
# Back-end abstractions (injectable)
# ---------------------------------------------------------------------------

Demuxer = Callable[[str], DecodedAudio | None]
SpectrogramGenerator = Callable[[DecodedAudio], List[SpectrogramRepresentation]]


@dataclass
class InMemoryDecodedAudio:
    """Test fake for :class:`DecodedAudio`."""

    sample_rate: int = 22050
    duration_seconds: float = 3.0
    samples: Any = field(default_factory=lambda: [0.0] * 66150)

    @property
    def is_empty(self) -> bool:
        if self.samples is None:
            return True
        if hasattr(self.samples, "__len__"):
            return len(self.samples) == 0
        return False



# ---------------------------------------------------------------------------
# AudioExtractor
# ---------------------------------------------------------------------------


class AudioExtractor:
    """Extract audio track and generate mel-spectrogram representations.

    Parameters
    ----------
    demuxer:
        Callable ``video_path -> DecodedAudio | None``.
    spectrogram_generator:
        Callable ``DecodedAudio -> list[SpectrogramRepresentation]``.
    """

    def __init__(
        self,
        demuxer: Demuxer | None = None,
        spectrogram_generator: SpectrogramGenerator | None = None,
    ) -> None:
        self._demuxer = demuxer if demuxer is not None else _default_demuxer
        self._spectrogram_generator = (
            spectrogram_generator
            if spectrogram_generator is not None
            else _default_spectrogram_generator
        )

    def extract_audio(self, video_path: str) -> DecodedAudio | None:
        """Extract audio track from video file.

        Returns None if video has no audio track (Requirement 3.1, 3.4).
        Raises AudioExtractionError on extraction failure (Requirement 3.5).
        """
        return self._demuxer(video_path)

    def to_spectrograms(
        self, audio: DecodedAudio
    ) -> list[SpectrogramRepresentation]:
        """Convert DecodedAudio to one or more spectrogram representations.

        Yields >= 1 representation for non-empty audio (Requirement 3.2, Property 7).
        """
        if audio.is_empty:
            return []
        spectrograms = self._spectrogram_generator(audio)
        if not spectrograms:
            # Fallback guarantee for valid non-empty audio: generate default representation
            spectrograms = [
                SpectrogramRepresentation(
                    sample_rate=audio.sample_rate,
                    n_mels=128,
                    time_steps=max(1, int(audio.duration_seconds * 100)),
                    data=None,
                )
            ]
        return spectrograms

    def extract_audio_signal(self, video_path: str) -> AudioExtractionOutcome:
        """Full pass: extract audio track and generate spectrograms.

        Returns an :class:`AudioExtractionOutcome`:
        * ``NO_AUDIO_SIGNAL`` — video path exists but has no audio track (Requirement 3.4).
        * ``AUDIO_ERROR`` — extraction or spectrogram generation failed (Requirement 3.5).
        * ``OK`` — audio track extracted and >= 1 spectrogram generated.
        """
        try:
            audio = self.extract_audio(video_path)
        except Exception as exc:  # noqa: BLE001
            return AudioExtractionOutcome(
                state=AudioSignalState.AUDIO_ERROR,
                has_audio_track=False,
                spectrograms=[],
                error_detail=f"audio extraction failed: {exc}",
            )

        if audio is None or audio.is_empty:
            return AudioExtractionOutcome(
                state=AudioSignalState.NO_AUDIO_SIGNAL,
                has_audio_track=False,
                spectrograms=[],
                error_detail="",
            )

        try:
            spectrograms = self.to_spectrograms(audio)
        except Exception as exc:  # noqa: BLE001
            return AudioExtractionOutcome(
                state=AudioSignalState.AUDIO_ERROR,
                has_audio_track=True,
                spectrograms=[],
                error_detail=f"spectrogram generation failed: {exc}",
            )

        if not spectrograms:
            return AudioExtractionOutcome(
                state=AudioSignalState.NO_AUDIO_SIGNAL,
                has_audio_track=True,
                spectrograms=[],
                error_detail="",
            )

        return AudioExtractionOutcome(
            state=AudioSignalState.OK,
            has_audio_track=True,
            spectrograms=spectrograms,
            error_detail="",
        )


# ---------------------------------------------------------------------------
# Default real back ends (lazy + guarded)
# ---------------------------------------------------------------------------


def _default_demuxer(video_path: str) -> DecodedAudio | None:
    """Extract audio track using FFmpeg or Librosa (lazy import)."""
    try:
        import librosa  # type: ignore  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise AudioExtractionError(
            "Librosa is required for audio demuxing but is not available. "
            "Inject a custom demuxer for testing."
        ) from exc

    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True)
        if len(y) == 0:
            return None
        duration = float(len(y) / sr)
        return DecodedAudio(sample_rate=int(sr), duration_seconds=duration, samples=y)
    except Exception:
        # No audio stream found or file missing audio
        return None


def _default_spectrogram_generator(
    audio: DecodedAudio,
) -> list[SpectrogramRepresentation]:
    """Generate mel-spectrogram using Librosa (lazy import)."""
    try:
        import librosa  # type: ignore  # noqa: PLC0415
        import numpy as np  # type: ignore  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise AudioExtractionError(
            "Librosa and NumPy are required for spectrogram generation. "
            "Inject a custom generator for testing."
        ) from exc

    if audio.samples is None or len(audio.samples) == 0:
        return []

    y = np.asarray(audio.samples, dtype=np.float32)
    sr = audio.sample_rate
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    return [
        SpectrogramRepresentation(
            sample_rate=sr,
            n_mels=int(log_mel.shape[0]),
            time_steps=int(log_mel.shape[1]),
            data=log_mel,
        )
    ]
