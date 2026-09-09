"""Dependency-free image helpers for XAI artifact rendering.

The XAI pipeline must produce three PNG artifacts per analyzed frame
(original frame, raw Grad-CAM heatmap, and the final overlay) without
requiring Pillow, OpenCV, or NumPy at import time. This module implements a
minimal, deterministic pure-Python pipeline:

* ``encode_png`` — a compact PNG encoder (zlib + struct, no third-party deps);
* ``jet`` — the classic jet colormap used for Grad-CAM heatmaps;
* ``nearest_upscale`` — deterministic nearest-neighbor upscaling;
* ``blend`` — alpha blending for the final overlay.

Everything here is pure Python so the XAI unit/property tests run in any
environment, and the artifacts are byte-deterministic for identical inputs.
"""

from __future__ import annotations

import struct
import zlib
from typing import Iterable, List, Sequence, Tuple

RGB = Tuple[int, int, int]
PixelRow = List[RGB]


# ---------------------------------------------------------------------------
# Minimal PNG encoder
# ---------------------------------------------------------------------------


def encode_png(width: int, height: int, rows: Sequence[PixelRow]) -> bytes:
    """Encode an RGB image as PNG bytes (pure Python, no dependencies).

    Parameters
    ----------
    width, height:
        Image dimensions. ``rows`` must have exactly ``height`` rows, each
        with exactly ``width`` ``(r, g, b)`` tuples in [0, 255].
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for row in rows:
        raw.append(0)  # filter type 0 (None) per scanline
        for r, g, b in row:
            raw.append(int(r) & 0xFF)
            raw.append(int(g) & 0xFF)
            raw.append(int(b) & 0xFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=6)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Jet colormap (standard piecewise-linear approximation)
# ---------------------------------------------------------------------------


def jet(value: float) -> RGB:
    """Map a normalized value in [0.0, 1.0] to an RGB jet color."""
    v = max(0.0, min(1.0, float(value)))
    r = max(0.0, min(1.0, 1.5 - abs(4.0 * v - 3.0)))
    g = max(0.0, min(1.0, 1.5 - abs(4.0 * v - 2.0)))
    b = max(0.0, min(1.0, 1.5 - abs(4.0 * v - 1.0)))
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def nearest_upscale(
    matrix: Sequence[Sequence[float]], target_width: int, target_height: int
) -> List[List[float]]:
    """Nearest-neighbor upscale a 2D matrix to ``(target_width, target_height)``."""
    src_h = len(matrix)
    src_w = len(matrix[0]) if src_h else 0
    if src_h == 0 or src_w == 0:
        return [[0.0] * target_width for _ in range(target_height)]
    out: List[List[float]] = []
    for y in range(target_height):
        sy = min(src_h - 1, int(y * src_h / target_height))
        row = matrix[sy]
        out_row: List[float] = []
        for x in range(target_width):
            sx = min(src_w - 1, int(x * src_w / target_width))
            out_row.append(row[sx])
        out.append(out_row)
    return out


def blend_rows(
    original: Sequence[PixelRow],
    heatmap: Sequence[PixelRow],
    alpha: float = 0.5,
) -> List[PixelRow]:
    """Alpha-blend ``heatmap`` over ``original`` (same dimensions)."""
    a = max(0.0, min(1.0, float(alpha)))
    out: List[PixelRow] = []
    for y, orig_row in enumerate(original):
        heat_row = heatmap[y] if y < len(heatmap) else orig_row
        blended: PixelRow = []
        for x, (or_, og, ob) in enumerate(orig_row):
            hr, hg, hb = heat_row[x] if x < len(heat_row) else (or_, og, ob)
            blended.append(
                (
                    int(round(or_ * (1.0 - a) + hr * a)),
                    int(round(og * (1.0 - a) + hg * a)),
                    int(round(ob * (1.0 - a) + hb * a)),
                )
            )
        out.append(blended)
    return out


def frame_to_rgb_rows(frame_image: object) -> Tuple[int, int, List[PixelRow]] | None:
    """Convert a frame image (numpy BGR array or nested lists) to RGB rows.

    Accepts:
    * ``numpy.ndarray`` with shape ``(H, W, 3)`` (BGR, as produced by OpenCV)
      or ``(H, W)`` grayscale;
    * nested ``list`` of ``(b, g, r)`` / ``(r, g, b)`` tuples or grayscale ints.

    Returns ``(width, height, rows)`` where rows are RGB pixel rows, or
    ``None`` if the input is not a recognized image.
    """
    if frame_image is None:
        return None

    # numpy array path (no import needed — duck-typing on shape/iteration)
    if hasattr(frame_image, "shape"):
        shape = tuple(frame_image.shape)
        if len(shape) == 3 and shape[2] >= 3:
            height, width = int(shape[0]), int(shape[1])
            rows: List[PixelRow] = []
            for y in range(height):
                row: PixelRow = []
                for x in range(width):
                    b, g, r = (
                        int(frame_image[y, x, 0]),
                        int(frame_image[y, x, 1]),
                        int(frame_image[y, x, 2]),
                    )
                    row.append((r, g, b))
                rows.append(row)
            return (width, height, rows)
        if len(shape) == 2:
            height, width = int(shape[0]), int(shape[1])
            gray_rows: List[PixelRow] = []
            for y in range(height):
                gray_row: PixelRow = []
                for x in range(width):
                    v = int(frame_image[y, x])
                    gray_row.append((v, v, v))
                gray_rows.append(gray_row)
            return (width, height, gray_rows)
        return None

    # nested-list path
    if isinstance(frame_image, (list, tuple)) and len(frame_image) > 0:
        first = frame_image[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            height = len(frame_image)
            width = len(first)
            list_rows: List[PixelRow] = []
            for y in range(height):
                list_row: PixelRow = []
                for x in range(width):
                    px = frame_image[y][x]
                    if isinstance(px, (list, tuple)) and len(px) >= 3:
                        r, g, b = int(px[0]), int(px[1]), int(px[2])
                    else:
                        v = int(px)
                        r = g = b = v
                    list_row.append((r, g, b))
                list_rows.append(list_row)
            return (width, height, list_rows)

    return None


def placeholder_original(width: int = 224, height: int = 224, gray: int = 32) -> List[PixelRow]:
    """Deterministic placeholder "original" frame (used when no frame is supplied)."""
    return [[(gray, gray, gray) for _ in range(width)] for _ in range(height)]


def render_artifacts(
    normalized_matrix: Sequence[Sequence[float]],
    frame_image: object = None,
    default_size: int = 224,
) -> Tuple[bytes, bytes, bytes]:
    """Render the (original, heatmap, overlay) PNG triple for one frame.

    Returns ``(original_png, heatmap_png, overlay_png)``. When ``frame_image``
    is unavailable a deterministic dark placeholder is used as the original.
    """
    converted = frame_to_rgb_rows(frame_image)
    if converted is not None:
        width, height, original_rows = converted
    else:
        width, height, original_rows = default_size, default_size, placeholder_original()

    target_w = max(1, int(width))
    target_h = max(1, int(height))

    upscaled = nearest_upscale(normalized_matrix, target_w, target_h)

    heat_rows: List[PixelRow] = []
    for y in range(target_h):
        heat_row: PixelRow = []
        for x in range(target_w):
            heat_row.append(jet(upscaled[y][x]))
        heat_rows.append(heat_row)

    overlay_rows = blend_rows(original_rows, heat_rows, alpha=0.5)

    return (
        encode_png(target_w, target_h, original_rows),
        encode_png(target_w, target_h, heat_rows),
        encode_png(target_w, target_h, overlay_rows),
    )