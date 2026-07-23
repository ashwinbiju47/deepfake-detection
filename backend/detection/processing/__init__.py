"""Media-processing components for the detection pipeline.

This package holds the pure, dependency-light processing logic (frame
sampling-rate computation, 5% face-area threshold, audio extraction,
spectrogram generation, and signal-state determination) separated from the
heavy OpenCV / MTCNN / FFmpeg / Librosa back ends so the logic stays unit-
and property-testable without the ML/CV/audio stack installed.
"""

from __future__ import annotations

from .audio_extractor import (
    AudioExtractionError,
    AudioExtractionOutcome,
    AudioExtractor,
    AudioSignalState,
    DecodedAudio,
    InMemoryDecodedAudio,
    SpectrogramRepresentation,
)
from .frame_extractor import (
    DecodedVideo,
    ExtractionOutcome,
    FaceCandidate,
    FaceRegion,
    Frame,
    FrameExtractor,
    InMemoryDecodedVideo,
    VisualSignalState,
)

__all__ = [
    "AudioExtractionError",
    "AudioExtractionOutcome",
    "AudioExtractor",
    "AudioSignalState",
    "DecodedAudio",
    "DecodedVideo",
    "ExtractionOutcome",
    "FaceCandidate",
    "FaceRegion",
    "Frame",
    "FrameExtractor",
    "InMemoryDecodedAudio",
    "InMemoryDecodedVideo",
    "SpectrogramRepresentation",
    "VisualSignalState",
]
