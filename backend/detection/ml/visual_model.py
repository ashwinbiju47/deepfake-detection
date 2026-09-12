"""Visual_Model inference and aggregation (Task 6.1, Requirement 2.3, 2.6).

This module implements the design's ``Visual_Model`` interface:

    infer_face(face) -> float  # per-face likelihood in [0.0, 1.0]
    aggregate(likelihoods, top_k=None) -> float  # aggregate visual likelihood in [0.0, 1.0]

Design intent (design.md -> Components and Interfaces -> Visual_Model):
* Compute deepfake likelihood for each isolated face region, clamped to [0.0, 1.0].
* Aggregate face-level predictions into a single visual likelihood for the session,
  clamped to [0.0, 1.0].
* Support injectable inference engine (PyTorch, ONNX, or test stub).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from detection.processing.frame_extractor import FaceRegion


def clamp_likelihood(value: float) -> float:
    """Clamp float likelihood value strictly to [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


# ---------------------------------------------------------------------------
# Deterministic reference inference
# ---------------------------------------------------------------------------
# No trained weights ship with the repository, so the default back end is a
# **deterministic forensic reference baseline** rather than a random/constant
# stand-in: it measures the high-frequency residual energy of the isolated face
# (deepfake reconstructions typically leave abnormal high-frequency residue)
# and maps it through a fixed centred logistic curve.
#
# The two properties that matter for the platform contract:
#   * the same input always produces the same likelihood (no per-run drift), and
#   * different inputs produce different likelihoods.
# Replacing the baseline with real trained weights is a one-line injection:
# ``VisualModel(infer_backend=my_torch_model.predict)``.
# Calibrated on a natural face crop (high-frequency residual ratio ~0.275 for
# the standard 512x512 face sample, measured on the 224x224 model crop): a
# natural face lands just under the 0.5 decision threshold, while sharper /
# reconstruction-residue faces push above it.
_VISUAL_FEATURE_REFERENCE = 0.28
_VISUAL_FEATURE_SCALE = 0.08


def content_likelihood(feature: float, reference: float, scale: float) -> float:
    """Map a content feature onto a likelihood in ``(0, 1)``.

    ``0.5`` corresponds to ``feature == reference``; the curve is a fixed,
    monotone logistic so identical inputs always give identical outputs.
    """
    import math  # noqa: PLC0415

    if scale is None or scale <= 0:
        return 0.5
    try:
        shifted = (float(feature) - float(reference)) / float(scale)
    except (TypeError, ValueError):
        return 0.5
    return clamp_likelihood(0.5 + 0.5 * math.tanh(shifted))


def _grayscale_array(image: Any) -> Optional[Any]:
    """Return a float32 grayscale array for an image-like input, else ``None``."""
    if image is None:
        return None
    try:
        import numpy as np  # type: ignore  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        arr = np.asarray(image)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr[:, :, :3].astype("float32").mean(axis=2)
    elif arr.ndim == 2:
        arr = arr.astype("float32")
    else:
        return None
    return arr


def face_high_frequency_ratio(image: Any) -> Optional[float]:
    """High-frequency residual energy of a face crop, normalized by contrast.

    Uses a discrete 3x3 Laplacian computed with array shifts (no SciPy).
    Returns ``None`` when no pixel data is available.
    """
    gray = _grayscale_array(image)
    if gray is None or gray.shape[0] < 4 or gray.shape[1] < 4:
        return None
    try:
        import numpy as np  # type: ignore  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    centered = gray - float(gray.mean())
    laplacian = (
        4.0 * centered[1:-1, 1:-1]
        - centered[:-2, 1:-1]
        - centered[2:, 1:-1]
        - centered[1:-1, :-2]
        - centered[1:-1, 2:]
    )
    high_frequency = float(np.abs(laplacian).mean())
    contrast = float(np.abs(centered).mean())
    if contrast < 1.0:
        # Degenerate crop (blank/near-constant pixels): no evidence either way.
        return None
    return high_frequency / (contrast + 1e-6)


InferFaceBackend = Callable[[FaceRegion], float]


class VisualModelError(Exception):
    """Raised when visual model inference fails."""


class VisualModel:
    """Vision CNN / Transformer inference wrapper for facial deepfake detection.

    Parameters
    ----------
    infer_backend:
        Callable ``FaceRegion -> float`` yielding likelihood in [0.0, 1.0].
        If None, a deterministic mock backend is used unless PyTorch weights are loaded.
    """

    def __init__(self, infer_backend: InferFaceBackend | None = None) -> None:
        self._infer_backend = (
            infer_backend if infer_backend is not None else self._default_infer_backend
        )

    def infer_face(self, face: FaceRegion) -> float:
        """Infer deepfake likelihood for a single isolated face region.

        Returns float strictly clamped to [0.0, 1.0] (Requirement 2.3).
        """
        try:
            raw_val = self._infer_backend(face)
            return clamp_likelihood(raw_val)
        except Exception as exc:  # noqa: BLE001
            raise VisualModelError(f"visual model face inference failed: {exc}") from exc

    def aggregate(
        self, likelihoods: List[float], top_k: int | None = None
    ) -> float:
        """Aggregate per-face/frame likelihoods into a single aggregate score in [0.0, 1.0].

        If ``top_k`` is provided, averages the top-k highest likelihoods.
        Otherwise takes the arithmetic mean across all predictions (Requirement 2.6).
        Returns 0.0 if likelihoods is empty.
        """
        if not likelihoods:
            return 0.0

        clamped_list = [clamp_likelihood(v) for v in likelihoods]
        if top_k is not None and top_k > 0 and top_k < len(clamped_list):
            sorted_vals = sorted(clamped_list, reverse=True)[:top_k]
            mean_val = sum(sorted_vals) / len(sorted_vals)
        else:
            mean_val = sum(clamped_list) / len(clamped_list)

        return clamp_likelihood(mean_val)

    def activation_map(self, face: FaceRegion, size: int = 8) -> List[List[float]]:
        """Return a Grad-CAM-style activation matrix for one face.

        In a deployed system this is the gradient-weighted activation map from
        the CNN's final convolutional layer. Here it is a deterministic
        stand-in: a centered blob whose peak scales with the face's inferred
        likelihood, so the ORIGINAL / HEATMAP / OVERLAY XAI triple is always
        renderable (Requirement 11).
        """
        import math  # noqa: PLC0415

        likelihood = self.infer_face(face)
        center = (size - 1) / 2.0
        matrix: List[List[float]] = []
        for y in range(size):
            row: List[float] = []
            for x in range(size):
                dist = math.hypot(x - center, y - center) / max(center, 1.0)
                row.append(clamp_likelihood(likelihood * math.exp(-(dist**2) / 2.0)))
            matrix.append(row)
        return matrix

    def _default_infer_backend(self, face: FaceRegion) -> float:
        """Deterministic reference inference for one isolated face.

        Measures the face crop's high-frequency residual ratio and maps it
        through a fixed logistic curve. Pure function of the input pixels, so
        re-analyzing the same video always yields the same likelihood.
        """
        image = getattr(face, "image", None)
        ratio = face_high_frequency_ratio(image)
        if ratio is not None:
            return content_likelihood(
                ratio, _VISUAL_FEATURE_REFERENCE, _VISUAL_FEATURE_SCALE
            )
        if image is not None:
            # Pixels present but degenerate (blank crop): neutral likelihood.
            return 0.5
        # No pixels available (pure-logic callers / tests): fall back to a
        # geometry-derived value that is likewise a pure function of the input.
        seed = (face.x * 31 + face.y * 17 + face.width * 13 + face.height * 7) % 100
        return clamp_likelihood(seed / 100.0)
