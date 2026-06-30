"""Property-based test for the 5% face-area threshold (Task 4.3).

# Feature: deepfake-detection-platform, Property 4: Face isolation respects the 5% area threshold

Validates: Requirements 2.2

This is a pure-logic test: ``FrameExtractor.isolate_faces`` runs an injected
detector and must keep **exactly** those candidate facial regions whose area is
at least 5% of the frame area, and no others. No Django DB, OpenCV, or MTCNN is
required — a plain function standing in for the detector returns the generated
candidate boxes.
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from detection.processing.frame_extractor import (
    FaceCandidate,
    Frame,
    FrameExtractor,
)

MIN_AREA_RATIO = 0.05


@st.composite
def frame_and_candidates(draw):
    """Generate a frame size plus candidate boxes of varied sizes.

    The candidate list mixes boxes above, below, and (sometimes) exactly at the
    5% area threshold so the property exercises both sides of the boundary as
    well as the exact-boundary case itself.
    """
    frame_w = draw(st.integers(min_value=1, max_value=1000))
    frame_h = draw(st.integers(min_value=1, max_value=1000))

    # Candidate boxes spanning the full range of sizes: zero-area (always below),
    # tiny (below), and up to full-frame (always above). This guarantees the
    # generated lists contain candidates on both sides of the threshold.
    box = st.builds(
        FaceCandidate,
        x=st.integers(min_value=0, max_value=frame_w),
        y=st.integers(min_value=0, max_value=frame_h),
        width=st.integers(min_value=0, max_value=frame_w),
        height=st.integers(min_value=0, max_value=frame_h),
    )
    candidates = draw(st.lists(box, min_size=0, max_size=12))

    # Optionally inject a candidate whose area is *exactly* 5% of the frame area.
    # Choosing frame dimensions (4*cw, 5*ch) makes the frame area 20*(cw*ch), so a
    # (cw, ch) box has ratio cw*ch / (20*cw*ch) == 1/20 == 0.05 exactly. Such a
    # candidate sits on the inclusive boundary and MUST be kept.
    if draw(st.booleans()):
        cw = draw(st.integers(min_value=1, max_value=50))
        ch = draw(st.integers(min_value=1, max_value=50))
        frame_w, frame_h = 4 * cw, 5 * ch
        # Rebuild the other candidates to stay within the (possibly) new frame.
        candidates = [
            FaceCandidate(
                x=min(c.x, frame_w),
                y=min(c.y, frame_h),
                width=min(c.width, frame_w),
                height=min(c.height, frame_h),
            )
            for c in candidates
        ]
        candidates.append(FaceCandidate(x=0, y=0, width=cw, height=ch))

    return frame_w, frame_h, candidates


@given(data=frame_and_candidates())
@settings(max_examples=200)
def test_isolate_faces_respects_five_percent_threshold(data) -> None:
    """isolate_faces keeps exactly the candidates with area >= 5% of frame area."""
    frame_w, frame_h, candidates = data
    frame = Frame(index=0, timestamp=0.0, width=frame_w, height=frame_h)
    frame_area = frame_w * frame_h

    # Inject a detector that returns the generated candidate boxes for this frame.
    extractor = FrameExtractor(detector=lambda _frame, _c=candidates: list(_c))

    regions = extractor.isolate_faces(frame, min_area_ratio=MIN_AREA_RATIO)

    # Expected: exactly those candidates whose area fraction is >= 5% (inclusive).
    expected_boxes = Counter(
        (c.x, c.y, c.width, c.height)
        for c in candidates
        if (c.width * c.height) / frame_area >= MIN_AREA_RATIO
    )
    actual_boxes = Counter(
        (r.x, r.y, r.width, r.height) for r in regions
    )

    # Exactly the qualifying candidates are kept — no others, none missing.
    assert actual_boxes == expected_boxes

    # Every returned region independently satisfies the threshold, and each
    # carries the originating frame's geometry through unchanged.
    for region in regions:
        assert region.area_ratio >= MIN_AREA_RATIO
        assert region.frame_index == frame.index
        assert region.frame_width == frame_w
        assert region.frame_height == frame_h
