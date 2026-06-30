"""Property 3: Frame extraction meets the minimum sampling rate (Requirement 2.1).

This is a pure-logic property test. It exercises ``FrameExtractor.extract_frames``
against an in-memory decoder fake (:class:`InMemoryDecodedVideo`) so no OpenCV,
no real video file, and no Django/DB are required.

The acceptance criterion (Requirement 2.1) requires the ``Frame_Extractor`` to
sample at **>= 1 frame per second of video duration**. For a video of duration
``d`` seconds that means it must yield at least ``floor(d)`` frames.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from detection.processing.frame_extractor import (
    Frame,
    FrameExtractor,
    InMemoryDecodedVideo,
)

# Generators constrain to the realistic input space (design Frame_Extractor):
#   * duration: 1..600 seconds, integer or float
#   * native frame_rate: 1..120 fps, integer or float
_durations = st.one_of(
    st.integers(min_value=1, max_value=600),
    st.floats(
        min_value=1.0,
        max_value=600.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_frame_rates = st.one_of(
    st.integers(min_value=1, max_value=120),
    st.floats(
        min_value=1.0,
        max_value=120.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)


def _build_decoded_video(duration: float, frame_rate: float) -> InMemoryDecodedVideo:
    """Build an in-memory decoded video with round(duration*frame_rate) raw frames."""
    raw_count = round(duration * frame_rate)
    frames = [
        Frame(index=i, timestamp=i / frame_rate, width=64, height=64)
        for i in range(raw_count)
    ]
    return InMemoryDecodedVideo(
        duration_seconds=float(duration),
        frame_rate=float(frame_rate),
        frames=frames,
    )


# Feature: deepfake-detection-platform, Property 3: Frame extraction meets the minimum sampling rate
@given(duration=_durations, frame_rate=_frame_rates)
@settings(max_examples=100)
def test_frame_extraction_meets_minimum_sampling_rate(
    duration: float, frame_rate: float
) -> None:
    """extract_frames yields at least floor(duration) frames (>= 1 fps of duration).

    **Validates: Requirements 2.1**
    """
    decoded = _build_decoded_video(duration, frame_rate)
    extractor = FrameExtractor(decoder=lambda _path: decoded)

    produced = list(extractor.extract_frames("in-memory://video", min_fps=1.0))

    expected_minimum = math.floor(duration)
    assert len(produced) >= expected_minimum, (
        f"duration={duration}, frame_rate={frame_rate}: "
        f"produced {len(produced)} frames, expected at least {expected_minimum}"
    )
