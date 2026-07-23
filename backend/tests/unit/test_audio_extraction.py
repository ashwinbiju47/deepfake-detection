"""Unit tests for AudioExtractor edge cases (Task 5.3, Requirements 3.1, 3.4, 3.5).

Tests audio extraction success, no-audio-track path, and exception handling paths.
"""

from __future__ import annotations

import pytest

from detection.processing.audio_extractor import (
    AudioExtractionError,
    AudioExtractor,
    AudioSignalState,
    DecodedAudio,
    InMemoryDecodedAudio,
    SpectrogramRepresentation,
)

pytestmark = pytest.mark.unit


def test_audio_extraction_success_with_audio_track() -> None:
    """Requirement 3.1 & 3.2: Valid audio track produces OK state and spectrograms."""
    audio = InMemoryDecodedAudio(sample_rate=22050, duration_seconds=2.0, samples=[0.0] * 44100)
    spec = SpectrogramRepresentation(sample_rate=22050, n_mels=128, time_steps=200, data=None)

    extractor = AudioExtractor(
        demuxer=lambda _path: audio,
        spectrogram_generator=lambda _audio: [spec],
    )

    outcome = extractor.extract_audio_signal("sample.mp4")
    assert outcome.state is AudioSignalState.OK
    assert outcome.has_audio_track is True
    assert len(outcome.spectrograms) == 1
    assert outcome.error_detail == ""
    assert outcome.has_audio_signal is True


def test_audio_extraction_no_audio_track() -> None:
    """Requirement 3.4: Video with no audio track yields NO_AUDIO_SIGNAL."""
    extractor = AudioExtractor(
        demuxer=lambda _path: None,
    )

    outcome = extractor.extract_audio_signal("silent.mp4")
    assert outcome.state is AudioSignalState.NO_AUDIO_SIGNAL
    assert outcome.has_audio_track is False
    assert outcome.spectrograms == []
    assert outcome.error_detail == ""
    assert outcome.has_audio_signal is False


def test_audio_extraction_failure_before_demux() -> None:
    """Requirement 3.5: Demux failure yields AUDIO_ERROR with error_detail."""
    def raising_demuxer(_path: str) -> DecodedAudio | None:
        raise AudioExtractionError("ffmpeg binary not found")

    extractor = AudioExtractor(demuxer=raising_demuxer)

    outcome = extractor.extract_audio_signal("corrupt.mp4")
    assert outcome.state is AudioSignalState.AUDIO_ERROR
    assert outcome.has_audio_track is False
    assert outcome.spectrograms == []
    assert "ffmpeg binary not found" in outcome.error_detail
    assert outcome.has_audio_signal is False


def test_audio_extraction_failure_during_spectrogram() -> None:
    """Requirement 3.5: Spectrogram generation failure yields AUDIO_ERROR."""
    audio = InMemoryDecodedAudio(sample_rate=22050, duration_seconds=2.0, samples=[0.0] * 44100)

    def raising_generator(_audio: DecodedAudio) -> list[SpectrogramRepresentation]:
        raise RuntimeError("librosa failed")

    extractor = AudioExtractor(
        demuxer=lambda _path: audio,
        spectrogram_generator=raising_generator,
    )

    outcome = extractor.extract_audio_signal("video.mp4")
    assert outcome.state is AudioSignalState.AUDIO_ERROR
    assert outcome.has_audio_track is True
    assert outcome.spectrograms == []
    assert "librosa failed" in outcome.error_detail
    assert outcome.has_audio_signal is False
