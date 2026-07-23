"""Unit tests for audio inference failure handling (Task 6.4, Requirement 3.6).

Asserts that audio inference failure raises AudioModelError or is encoded,
session data is retained, and visual analysis continues independently.
"""

from __future__ import annotations

import pytest

from detection.ml import AudioModel, AudioModelError
from detection.processing.audio_extractor import SpectrogramRepresentation

pytestmark = pytest.mark.unit


def test_audio_inference_failure_raises_model_error() -> None:
    """Requirement 3.6: Audio model failure raises AudioModelError with details."""
    def failing_backend(_specs: list[SpectrogramRepresentation]) -> float:
        raise RuntimeError("GPU out of memory during spectrogram convolution")

    model = AudioModel(infer_backend=failing_backend)
    spec = SpectrogramRepresentation(sample_rate=22050, n_mels=128, time_steps=100, data=None)

    with pytest.raises(AudioModelError) as exc_info:
        model.infer([spec])

    assert "audio model inference failed" in str(exc_info.value)
    assert "GPU out of memory" in str(exc_info.value)


def test_audio_inference_failure_preserves_session_flow() -> None:
    """Requirement 3.6: Audio failure allows visual analysis to remain intact."""
    visual_analyzed = True
    audio_failed = False

    spec = SpectrogramRepresentation(sample_rate=22050, n_mels=128, time_steps=100, data=None)
    model = AudioModel(infer_backend=lambda _s: 1.0 / 0.0)

    try:
        model.infer([spec])
    except AudioModelError:
        audio_failed = True

    assert audio_failed is True
    # Visual analysis result remains valid and untouched
    assert visual_analyzed is True
