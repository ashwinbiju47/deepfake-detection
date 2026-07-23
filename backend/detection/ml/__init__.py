"""Machine Learning inference modules for visual and audio analysis.
"""

from __future__ import annotations

from .audio_model import AudioModel, AudioModelError
from .visual_model import VisualModel, VisualModelError, clamp_likelihood

__all__ = [
    "AudioModel",
    "AudioModelError",
    "VisualModel",
    "VisualModelError",
    "clamp_likelihood",
]
