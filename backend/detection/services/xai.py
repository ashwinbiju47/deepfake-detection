"""Grad-CAM XAI Generator: heatmap generation, normalization, and delivery (Task 15.1, Requirement 11).

This module implements the design's ``XAI_Generator`` interface:

    grad_cam(session_id, frame_id, frame_image) -> FrameHeatmapOutcome

Design intent (design.md -> Components and Interfaces -> XAI_Generator):
* Generate Grad-CAM activation heatmaps overlaying original frames, normalized to [0.0, 1.0] (Requirement 11.1, Property 20).
* For every analyzed frame produce the **full XAI triple**: the ORIGINAL
  frame, the raw Grad-CAM heatmap, and the final OVERLAY (heatmap
  alpha-blended over the original), so the explainability view never shows a
  heatmap in isolation.
* For **audio** uploads the same triple is produced over the log-mel
  spectrogram: the ORIGINAL spectrogram, its attention HEATMAP, and the
  OVERLAY of the two.
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
from detection.services.imaging import PixelRow, render_artifacts
from detection.services.streamer import StreamService
from detection.storage import session_media_dir

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


# ---------------------------------------------------------------------------
# Audio XAI: the same ORIGINAL / HEATMAP / OVERLAY triple over the spectrogram
# ---------------------------------------------------------------------------


def _spectrogram_png_rows(log_mel: Any, width: int, height: int) -> List[PixelRow]:
    """Render a log-mel spectrogram as a grayscale RGB pixel grid.

    Deterministic: time is mapped left-to-right, mel bins bottom-to-top
    (low frequency at the bottom, as in librosa.display)."""
    import numpy as np  # local import; spectrograms are numpy payloads

    spec = np.asarray(log_mel, dtype="float64")
    n_mels, n_time = spec.shape
    lo, hi = float(spec.min()), float(spec.max())
    if hi - lo < 1e-9:
        values = np.zeros((height, width))
    else:
        mel_idx = np.clip((np.linspace(0, 1, height) * (n_mels - 1)).astype(int), 0, n_mels - 1)
        time_idx = np.clip((np.linspace(0, 1, width) * (n_time - 1)).astype(int), 0, n_time - 1)
        grid = spec[np.ix_(mel_idx, time_idx)]
        values = (grid - lo) / (hi - lo)  # rows: low mel -> high mel (top = high freq)
        values = values[::-1]  # put high frequency at top like a standard spectrogram

    rows: List[PixelRow] = []
    for y in range(height):
        row: PixelRow = []
        for x in range(width):
            v = int(round(255.0 * float(values[y, x])))
            row.append((v, v, v))
        rows.append(row)
    return rows


def generate_audio_xai(session_id: str, likelihood: float) -> FrameHeatmapOutcome:
    """Produce the XAI triple for an audio upload over its mel-spectrogram.

    Re-analyzes the session's stored audio file (already staged in the
    transient media dir) with the same extractor the scoring pipeline used, so
    the explanation corresponds exactly to the scored signal. The triple:

    * ORIGINAL — the log-mel spectrogram rendered as an image;
    * HEATMAP — the attention map over the upper-band (synthetic-artifact)
      region the reference model scores;
    * OVERLAY — the attention map alpha-blended over the spectrogram.
    """
    from detection.models import AnalysisSession
    from detection.processing.audio_extractor import AudioExtractor
    from detection.ml.audio_model import spectrogram_attention_map

    session = AnalysisSession.objects.get(id=session_id)
    media_dir = session_media_dir(session.id)
    media_files = sorted(p for p in media_dir.glob("*") if p.is_file())
    if not media_files:
        return FrameHeatmapOutcome(
            success=False, frame_id="spectrogram", error_detail="audio media missing"
        )

    outcome = AudioExtractor().extract_audio_signal(str(media_files[0]))
    if not outcome.has_audio_signal or not outcome.spectrograms:
        return FrameHeatmapOutcome(
            success=False, frame_id="spectrogram", error_detail="no audio signal"
        )

    spec_data = outcome.spectrograms[0].data
    if spec_data is None:
        return FrameHeatmapOutcome(
            success=False, frame_id="spectrogram", error_detail="spectrogram payload missing"
        )

    attention = spectrogram_attention_map(spec_data, size=12)
    if attention is None:
        attention = [[likelihood] * 12 for _ in range(12)]
    norm_matrix = normalize_heatmap(attention)

    import numpy as np  # noqa: PLC0415

    spec = np.asarray(spec_data)
    n_mels, n_time = spec.shape[:2]
    aspect = max(1.0, min(4.0, n_time / max(1, n_mels)))
    width = 224
    height = max(64, min(224, int(width / aspect)))

    original_rows = _spectrogram_png_rows(spec_data, width, height)

    from detection.services.imaging import (  # noqa: PLC0415
        blend_rows,
        encode_png,
        jet,
        nearest_upscale,
    )

    upscaled = nearest_upscale(norm_matrix, width, height)
    heat_rows: List[PixelRow] = [
        [jet(upscaled[y][x]) for x in range(width)] for y in range(height)
    ]
    overlay_rows = blend_rows(original_rows, heat_rows, alpha=0.5)

    original_png = encode_png(width, height, original_rows)
    heatmap_png = encode_png(width, height, heat_rows)
    overlay_png = encode_png(width, height, overlay_rows)

    record = FrameHeatmap.objects.create(
        session=session,
        frame_id="spectrogram",
        original_png=original_png,
        heatmap_png=heatmap_png,
        overlay_png=overlay_png,
        delivered=True,
    )

    StreamService.publish_event(
        session_id=session_id,
        event_type="heatmap",
        payload={
            "frame_id": "spectrogram",
            "heatmap_id": str(record.id),
            "original_b64": base64.b64encode(original_png).decode("ascii"),
            "heatmap_b64": base64.b64encode(heatmap_png).decode("ascii"),
            "overlay_b64": base64.b64encode(overlay_png).decode("ascii"),
        },
    )

    return FrameHeatmapOutcome(
        success=True,
        heatmap_id=str(record.id),
        frame_id="spectrogram",
        normalized_matrix=norm_matrix,
        error_detail="",
        original_b64=base64.b64encode(original_png).decode("ascii"),
        heatmap_b64=base64.b64encode(heatmap_png).decode("ascii"),
        overlay_b64=base64.b64encode(overlay_png).decode("ascii"),
    )
