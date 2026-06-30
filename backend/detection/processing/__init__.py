"""Media-processing components for the detection pipeline.

This package holds the pure, dependency-light processing logic (frame
sampling-rate computation, the 5% face-area threshold, signal-state
determination) separated from the heavy OpenCV / MTCNN / FFmpeg back ends so the
logic stays unit- and property-testable without the ML/CV stack installed.
"""

from __future__ import annotations

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
    "DecodedVideo",
    "ExtractionOutcome",
    "FaceCandidate",
    "FaceRegion",
    "Frame",
    "FrameExtractor",
    "InMemoryDecodedVideo",
    "VisualSignalState",
]
