"""Property 7: Spectrogram generation yields at least one representation (Task 5.2, Requirement 3.2).

Feature: deepfake-detection-platform, Property 7: Spectrogram generation yields at least one representation

For any valid, non-empty audio input, the Audio_Extractor MUST generate at least
one spectrogram representation.
"""

from __future__ import annotations

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from detection.processing.audio_extractor import (
    AudioExtractor,
    DecodedAudio,
    InMemoryDecodedAudio,
)

pytestmark = pytest.mark.property


@given(
    sample_rate=st.integers(min_value=8000, max_value=48000),
    duration=st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False),
    num_samples=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=100)
def test_property_spectrogram_generation_yields_at_least_one_representation(
    sample_rate: int, duration: float, num_samples: int
) -> None:
    """Property 7: Non-empty audio input yields >= 1 spectrogram representation."""
    samples = [0.1] * num_samples
    audio = DecodedAudio(
        sample_rate=sample_rate,
        duration_seconds=duration,
        samples=samples,
    )

    extractor = AudioExtractor(
        demuxer=lambda _path: audio,
        spectrogram_generator=lambda _audio: [],  # Fallback should kick in
    )

    spectrograms = extractor.to_spectrograms(audio)
    assert len(spectrograms) >= 1, "Spectrogram generation must yield at least one representation"
    assert spectrograms[0].sample_rate == sample_rate
    assert spectrograms[0].n_mels > 0
    assert spectrograms[0].time_steps > 0
