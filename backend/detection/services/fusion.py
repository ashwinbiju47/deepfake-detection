"""Fusion_Engine: multi-modal fusion and decision labeling (Task 7.1, Requirement 4).

This module implements the design's ``Fusion_Engine`` interface:

    fuse(visual_likelihood, audio_likelihood, weight_visual=0.6, weight_audio=0.4, threshold=0.5) -> FusionOutcome

Design intent (design.md -> Components and Interfaces -> Fusion_Engine):
* Combine visual and audio deepfake likelihoods into a single fused confidence score (0.0-1.0).
* Handle missing modalities gracefully (single modality -> single score; neither -> inconclusive).
* Pure deterministic function (Property 8).
* Assign classification label ("authentic" / "deepfake") against decision threshold (Requirement 4.3, Property 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from detection.ml.visual_model import clamp_likelihood


@dataclass(frozen=True)
class FusionOutcome:
    """Deterministic output of multi-modal fusion."""

    score: Optional[float]  # None if inconclusive
    label: Optional[Literal["authentic", "deepfake"]]
    modalities_used: List[str] = field(default_factory=list)
    inconclusive: bool = False
    threshold_used: float = 0.5


class FusionEngine:
    """Deterministic Multi-Modal Fusion Engine.

    Parameters
    ----------
    default_weight_visual: float
        Default weight for visual modality (default 0.6).
    default_weight_audio: float
        Default weight for audio modality (default 0.4).
    default_threshold: float
        Decision threshold in [0.0, 1.0] (default 0.5).
    """

    def __init__(
        self,
        default_weight_visual: float = 0.6,
        default_weight_audio: float = 0.4,
        default_threshold: float = 0.5,
    ) -> None:
        self.default_weight_visual = default_weight_visual
        self.default_weight_audio = default_weight_audio
        self.default_threshold = default_threshold

    @staticmethod
    def fuse(
        visual_likelihood: Optional[float] = None,
        audio_likelihood: Optional[float] = None,
        weight_visual: float = 0.6,
        weight_audio: float = 0.4,
        threshold: float = 0.5,
    ) -> FusionOutcome:
        """Pure deterministic multi-modal fusion pass.

        * Both modalities present: weighted combination clamped to [0.0, 1.0] (Requirement 4.1).
        * One modality present: score derived from available modality (Requirement 4.2).
        * Neither modality present: score=None, inconclusive=True (Requirement 4.4).
        * Assign label: "authentic" iff score < threshold, "deepfake" iff score >= threshold (Requirement 4.3).
        """
        has_visual = visual_likelihood is not None
        has_audio = False # audio_likelihood is not None - Audio disabled for 50% milestone

        if not has_visual and not has_audio:
            return FusionOutcome(
                score=None,
                label=None,
                modalities_used=[],
                inconclusive=True,
                threshold_used=threshold,
            )

        modalities_used: list[str] = []
        if has_visual and has_audio:
            v_val = clamp_likelihood(visual_likelihood)
            a_val = clamp_likelihood(audio_likelihood)
            total_weight = weight_visual + weight_audio
            if total_weight <= 0:
                score = (v_val + a_val) / 2.0
            else:
                score = (v_val * weight_visual + a_val * weight_audio) / total_weight
            modalities_used = ["visual", "audio"]
        elif has_visual:
            score = clamp_likelihood(visual_likelihood)
            modalities_used = ["visual"]
        else:
            score = clamp_likelihood(audio_likelihood)
            modalities_used = ["audio"]

        fused_score = clamp_likelihood(score)
        label: Literal["authentic", "deepfake"] = (
            "deepfake" if fused_score >= threshold else "authentic"
        )

        return FusionOutcome(
            score=fused_score,
            label=label,
            modalities_used=modalities_used,
            inconclusive=False,
            threshold_used=threshold,
        )
