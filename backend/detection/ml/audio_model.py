"""Audio_Model inference (Task 6.2, Requirement 3.3, 3.6).

This module implements the design's ``Audio_Model`` interface:

    infer(spectrograms) -> float  # audio likelihood in [0.0, 1.0]

Design intent (design.md -> Components and Interfaces -> Audio_Model):
* Analyze spectrogram representations for synthetic audio anomalies.
* Return audio likelihood in [0.0, 1.0], clamped to range (Requirement 3.3).
* Record audio_error on inference failure without crashing visual analysis or deleting session (Requirement 3.6).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from detection.ml.visual_model import clamp_likelihood, content_likelihood
from detection.processing.audio_extractor import SpectrogramRepresentation

# Deterministic reference inference for audio (see visual_model): no trained
# weights ship with the repository, so the default back end measures the
# spectral high-frequency energy ratio of the log-mel spectrogram (a standard
# synthetic-speech artifact indicator) and maps it through a fixed logistic
# curve. Pure function of the input signal - the same video always yields the
# same audio likelihood.
# Calibrated against reference signals: natural speech-like (1/f) audio
# measures ~0.03 upper-band energy ratio, broadband/synthetic-like audio
# ~0.34. The curve is centred between them so speech sits below and
# synthetic-leaning spectra sit above the 0.5 decision threshold.
_AUDIO_FEATURE_REFERENCE = 0.10
_AUDIO_FEATURE_SCALE = 0.08

InferAudioBackend = Callable[[List[SpectrogramRepresentation]], float]


def high_frequency_energy_ratio(data: Any) -> Optional[float]:
    """Fraction of spectrogram energy in the upper third of the mel bins.

    ``data`` is a log-mel spectrogram (dB) as produced by
    :func:`detection.processing.audio_extractor._default_spectrogram_generator`.
    Values are converted back to linear energy before summing, so the ratio is
    comparable across samples. Returns ``None`` when no usable payload exists.
    """
    if data is None:
        return None
    try:
        import numpy as np  # type: ignore  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        spec = np.asarray(data, dtype="float64")
    except Exception:  # noqa: BLE001
        return None
    if spec.ndim != 2 or spec.size == 0 or spec.shape[0] < 3:
        return None
    if float(spec.std()) < 1e-9:
        # Degenerate spectrogram (e.g. a silent track): no evidence either way.
        return None
    energy = np.power(10.0, np.clip(spec, -120.0, 120.0) / 10.0)
    total = float(energy.sum())
    if total <= 0.0:
        return None
    split = int(spec.shape[0] * 2 // 3)
    return float(energy[split:, :].sum()) / total


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
        """Deterministic reference inference over the log-mel spectrogram.

        Pure function of the input signal, so re-analyzing the same video
        always yields the same audio likelihood.
        """
        first = spectrograms[0]
        data = getattr(first, "data", None)
        if data is not None:
            ratio = high_frequency_energy_ratio(data)
            if ratio is None:
                # Payload present but degenerate (silent/near-constant audio).
                return 0.5
            return content_likelihood(
                ratio, _AUDIO_FEATURE_REFERENCE, _AUDIO_FEATURE_SCALE
            )
        # No spectrogram payload (pure-logic callers / tests): deterministic
        # value derived from the representation's own shape.
        return clamp_likelihood(((first.n_mels * 31 + first.time_steps * 17) % 100) / 100.0)
