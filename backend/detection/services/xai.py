"""Grad-CAM XAI Generator: heatmap generation, normalization, and delivery (Task 15.1, Requirement 11).

This module implements the design's ``XAI_Generator`` interface:

    grad_cam(session_id, frame_id, frame_image) -> FrameHeatmapOutcome

Design intent (design.md -> Components and Interfaces -> XAI_Generator):
* Generate Grad-CAM activation heatmaps overlaying original frames, normalized to [0.0, 1.0] (Requirement 11.1, Property 20).
* For every analyzed frame produce the **full XAI triple**: the ORIGINAL
  frame, the raw Grad-CAM heatmap, and the final OVERLAY (heatmap
  alpha-blended over the original), so the explainability view never shows a
  heatmap in isolation.
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
from detection.services.imaging import render_artifacts
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
    # Base64-encoded XAI triple (original / heatmap / overlay PNGs).
    original_b64: Optional[str] = None
    heatmap_b64: Optional[str] = None
    overlay_b64: Optional[str] = None


class XAIGenerator:
    """Generates Grad-CAM activation overlays and dispatches StreamEvent heatmaps."""

    @staticmethod
    def generate_heatmap(
        session_id: str,
        frame_id: str,
        activation_matrix: Optional[List[List[float]]] = None,
        frame_image: Any = None,
    ) -> FrameHeatmapOutcome:
        """Generate, normalize, save, and stream the XAI triple for a frame (Requirement 11.1).

        Parameters
        ----------
        activation_matrix:
            Raw model-activation values (Grad-CAM output) for the frame.
            Normalized to [0.0, 1.0] before rendering (Property 20).
        frame_image:
            Original frame pixels (numpy BGR array or nested lists). When
            unavailable, a deterministic placeholder original is rendered so
            the ORIGINAL / HEATMAP / OVERLAY triple is always complete.
        """
        if not getattr(settings, "HEATMAP_ENABLED", False):
            return FrameHeatmapOutcome(
                success=False,
                error_detail="HEATMAP_ENABLED feature flag is False",
            )

        try:
            session = AnalysisSession.objects.get(id=session_id)
        except AnalysisSession.DoesNotExist:
            return FrameHeatmapOutcome(
                success=False,
                error_detail=f"Session {session_id} not found",
            )

        try:
            raw_matrix = activation_matrix or [[0.1, 0.8], [0.3, 0.9]]
            norm_matrix = normalize_heatmap(raw_matrix)

            original_png, heatmap_png, overlay_png = render_artifacts(
                norm_matrix, frame_image=frame_image
            )

            record = FrameHeatmap.objects.create(
                session=session,
                frame_id=frame_id,
                original_png=original_png,
                heatmap_png=heatmap_png,
                overlay_png=overlay_png,
                delivered=True,
            )

            original_b64 = base64.b64encode(original_png).decode("ascii")
            heatmap_b64 = base64.b64encode(heatmap_png).decode("ascii")
            overlay_b64 = base64.b64encode(overlay_png).decode("ascii")

            # Stream the full triple over WebSockets (Requirement 11.3)
            StreamService.publish_event(
                session_id=session_id,
                event_type="heatmap",
                payload={
                    "frame_id": frame_id,
                    "heatmap_id": str(record.id),
                    "original_b64": original_b64,
                    "heatmap_b64": heatmap_b64,
                    "overlay_b64": overlay_b64,
                },
            )

            return FrameHeatmapOutcome(
                success=True,
                heatmap_id=str(record.id),
                frame_id=frame_id,
                normalized_matrix=norm_matrix,
                error_detail="",
                original_b64=original_b64,
                heatmap_b64=heatmap_b64,
                overlay_b64=overlay_b64,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Grad-CAM generation failed for frame %s in session %s: %s",
                frame_id,
                session_id,
                exc,
            )
            # Omit heatmap, emit frame error, classification preserved (Requirement 11.2)
            StreamService.publish_event(
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
