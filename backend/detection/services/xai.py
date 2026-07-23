"""Grad-CAM XAI Generator: heatmap generation, normalization, and delivery (Task 15.1, Requirement 11).

This module implements the design's ``XAI_Generator`` interface:

    grad_cam(session_id, frame_id, frame_image) -> FrameHeatmapOutcome

Design intent (design.md -> Components and Interfaces -> XAI_Generator):
* Generate Grad-CAM activation heatmaps overlaying original frames, normalized to [0.0, 1.0] (Requirement 11.1, Property 20).
* Persist FrameHeatmap records as non-source-media visual artifacts.
* On generation failure, retain classification, omit heatmap, emit error for that frame (Requirement 11.2).
* Stream heatmaps over WebSockets to connected client (Requirement 11.3, 11.4).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, List, Optional
from django.conf import settings

from detection.models import AnalysisSession, FrameHeatmap
from detection.services.streamer import StreamService

logger = logging.getLogger(__name__)


def normalize_heatmap(matrix: List[List[float]]) -> List[List[float]]:
    """Normalize 2D floating-point activation matrix strictly into [0.0, 1.0]."""
    import math

    if not matrix or not matrix[0]:
        return []

    min_val = min(min(row) for row in matrix)
    max_val = max(max(row) for row in matrix)

    val_range = max_val - min_val
    if val_range <= 0.0 or math.isinf(val_range) or math.isnan(val_range):
        return [[0.0 for _ in row] for row in matrix]

    result: List[List[float]] = []
    for row in matrix:
        norm_row: List[float] = []
        for v in row:
            try:
                norm_v = (v - min_val) / val_range
                if math.isnan(norm_v) or norm_v < 0.0:
                    norm_v = 0.0
                elif norm_v > 1.0:
                    norm_v = 1.0
            except Exception:
                norm_v = 0.0
            norm_row.append(norm_v)
        result.append(norm_row)
    return result



@dataclass(frozen=True)
class FrameHeatmapOutcome:
    success: bool
    heatmap_id: Optional[str] = None
    frame_id: Optional[str] = None
    normalized_matrix: Optional[List[List[float]]] = None
    error_detail: str = ""


class XAIGenerator:
    """Generates Grad-CAM activation overlays and dispatches StreamEvent heatmaps."""

    @staticmethod
    def generate(
        session_id: str,
        frame_id: str,
        activation_matrix: Optional[List[List[float]]] = None,
    ) -> FrameHeatmapOutcome:
        """Generate, normalize, save, and stream Grad-CAM heatmap for a frame (Requirement 11.1)."""
        return FrameHeatmapOutcome(
            success=False,
            error_detail="HEATMAP disabled for 50% milestone",
                session_id=session_id,
                event_type="error",
                payload={
                    "error_code": "HEATMAP_GENERATION_FAILED",
                    "frame_id": frame_id,
                    "message": f"Grad-CAM failed for frame {frame_id}: {exc}",
                },
            )
            return FrameHeatmapOutcome(
                success=False,
                frame_id=frame_id,
                error_detail=str(exc),
            )
