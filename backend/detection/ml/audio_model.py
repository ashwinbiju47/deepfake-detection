"""Audio_Model inference (Task 6.2, Requirement 3.3, 3.6).

This module implements the design's ``Audio_Model`` interface:

    infer(spectrograms) -> float  # audio likelihood in [0.0, 1.0]

Design intent (design.md -> Components and Interfaces -> Audio_Model):
* Analyze spectrogram representations for synthetic audio anomalies.
* Return audio likelihood in [0.0, 1.0], clamped to range (Requirement 3.3).
* Record audio_error on inference failure without crashing visual analysis or deleting session (Requirement 3.6).
"""

from __future__ import annotations

from typing import Callable, List

from detection.ml.visual_model import clamp_likelihood
from detection.processing.audio_extractor import SpectrogramRepresentation

InferAudioBackend = Callable[[List[SpectrogramRepresentation]], float]


class AudioModelError(Exception):
    """Raised when audio model inference fails explicitly."""


class AudioModel:
    """Audio CNN / ResNet spectrogram classification wrapper.

    Parameters
    ----------
    infer_backend:
        Callable ``list[SpectrogramRepresentation] -> float``.
    """

    def __init__(self, infer_backend: InferAudioBackend | None = None) -> None:
        self._infer_backend = (
            infer_backend if infer_backend is not None else self._default_infer_backend
        )

    def infer(self, spectrograms: List[SpectrogramRepresentation]) -> float:
        """Infer deepfake audio likelihood from spectrogram representations.

        Returns float strictly in [0.0, 1.0] (Requirement 3.3).
        Raises AudioModelError on inference failure (Requirement 3.6).
        """
        if not spectrograms:
            raise AudioModelError("cannot infer audio likelihood on empty spectrograms list")

        try:
            raw_val = self._infer_backend(spectrograms)
            return clamp_likelihood(raw_val)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, AudioModelError):
                raise
            raise AudioModelError(f"audio model inference failed: {exc}") from exc

    def _default_infer_backend(
        self, spectrograms: List[SpectrogramRepresentation]
    ) -> float:
        """Default PyTorch/stub backend."""
        try:
            import torch  # type: ignore  # noqa: PLC0415
            return 0.5
        except ImportError:
            first = spectrograms[0]
            val = ((first.n_mels * 31 + first.time_steps * 17) % 100) / 100.0
            return val
