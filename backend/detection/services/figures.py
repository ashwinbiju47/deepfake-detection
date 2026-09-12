"""Evaluation figures for the Results / Evaluation chapter.

Renders the four figures that accompany the benchmark tables:

1. **Confusion matrices** - one panel per modality configuration
   (multimodal / visual-only / audio-only), annotated with TP / TN / FP / FN
   cell meanings so the two error types are readable at a glance.
2. **ROC curves** - all three models on one axes with their AUCs.
3. **Precision-Recall curves** - all three models on one axes with their
   average precision.
4. **Fusion-weight ablation** - accuracy and ROC-AUC against the visual fusion
   weight, with the configured 0.6 setting marked.

Everything is derived from :mod:`detection.services.benchmark` and
:mod:`detection.services.ablation`, so the figures cannot drift from the
tables. PNGs are returned base64-encoded for JSON transport (dashboard) and
raw for PDF embedding.

Matplotlib is imported lazily and guarded: if it is unavailable the figures are
simply omitted (``None``), and the tables remain served.
"""

from __future__ import annotations

import base64
import io
import threading
from typing import Any, Dict, List, Optional

from detection.services.ablation import ablation_table
from detection.services.benchmark import (
    MODALITY_COMPARISON,
    MODALITY_ORDERING,
    VARIANT_LABEL,
    curves_payload,
)

# One distinct colour per modality, consistent across every figure.
_VARIANT_COLORS: Dict[str, str] = {
    "multimodal": "#0f766e",
    "visual_only": "#2563eb",
    "audio_only": "#b45309",
}

_CACHE: Dict[str, Optional[bytes]] = {}
_CACHE_LOCK = threading.Lock()


def _new_figure(figsize: tuple[float, float], ncols: int = 1) -> Any:
    """Create a matplotlib figure and its axes on the Agg backend."""
    import matplotlib  # type: ignore  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore  # noqa: PLC0415

    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=140)
    return fig, axes


def _to_png_bytes(fig: Any) -> bytes:
    """Serialize a figure to PNG bytes and close it."""
    import matplotlib.pyplot as plt  # type: ignore  # noqa: PLC0415

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def confusion_matrix_figure() -> Optional[bytes]:
    """Three annotated confusion-matrix panels, one per modality configuration."""
    try:
        fig, axes = _new_figure((13.0, 4.4), ncols=3)
    except Exception:  # noqa: BLE001 - matplotlib unavailable
        return None

    rows = {r.variant: r for r in MODALITY_COMPARISON}
    for ax, variant in zip(axes, MODALITY_ORDERING):
        row = rows[variant]
        matrix = [[row.tn, row.fp], [row.fn, row.tp]]
        # Row order: actual authentic / actual deepfake; col: predicted auth/fake.
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix[0] + matrix[1]) or 1)
        ax.set_xticks([0, 1], labels=["predicted\nauthentic", "predicted\ndeepfake"])
        ax.set_yticks([0, 1], labels=["actually\nauthentic", "actually\ndeepfake"])
        ax.set_title(
            f"{VARIANT_LABEL[variant]}\naccuracy {row.cm.accuracy * 100:.1f}%",
            fontsize=10,
            fontweight="bold",
            color=_VARIANT_COLORS[variant],
        )
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels[i][j]}\n{matrix[i][j]}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="#111827",
                )
        ax.set_xlabel("model prediction", fontsize=8)
        if variant == "multimodal":
            ax.set_ylabel("ground truth", fontsize=8)
    fig.suptitle(
        "Confusion matrices per modality — FP = authentic video wrongly flagged "
        "as deepfake, FN = deepfake missed",
        fontsize=10,
    )
    return _to_png_bytes(fig)


def roc_figure() -> Optional[bytes]:
    """Single ROC axes containing multimodal, visual-only and audio-only curves."""
    try:
        fig, ax = _new_figure((6.6, 5.6))
    except Exception:  # noqa: BLE001 - matplotlib unavailable
        return None

    payload = curves_payload()
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1, label="chance (AUC 0.500)")
    for curve in payload["roc"]:
        variant = str(curve["variant"])
        points = curve["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            linewidth=2,
            color=_VARIANT_COLORS[variant],
            label=f"{VARIANT_LABEL[variant]}  (AUC {curve['auc']:.3f})",
        )
        op = curve["operating_point"]
        ax.plot(
            [op["fpr"]],
            [op["tpr"]],
            marker="o",
            markersize=5,
            color=_VARIANT_COLORS[variant],
        )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate (1 − specificity)")
    ax.set_ylabel("True Positive Rate (recall)")
    ax.set_title("ROC curves — all three models", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    return _to_png_bytes(fig)


def pr_figure() -> Optional[bytes]:
    """Single Precision-Recall axes for all three models."""
    try:
        fig, ax = _new_figure((6.6, 5.6))
    except Exception:  # noqa: BLE001 - matplotlib unavailable
        return None

    payload = curves_payload()
    row = MODALITY_COMPARISON[0]
    prevalence = (row.tp + row.fn) / max(1, row.tp + row.fn + row.fp + row.tn)
    ax.axhline(
        prevalence,
        linestyle="--",
        color="#9ca3af",
        linewidth=1,
        label=f"chance (prevalence {prevalence:.2f})",
    )
    for curve in payload["pr"]:
        variant = str(curve["variant"])
        points = curve["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            linewidth=2,
            color=_VARIANT_COLORS[variant],
            label=(
                f"{VARIANT_LABEL[variant]}  "
                f"(AP {curve['average_precision']:.3f})"
            ),
        )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — all three models", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    return _to_png_bytes(fig)


def ablation_figure() -> Optional[bytes]:
    """Accuracy / ROC-AUC against the visual fusion weight."""
    try:
        fig, ax = _new_figure((6.8, 4.6))
    except Exception:  # noqa: BLE001 - matplotlib unavailable
        return None

    table = ablation_table()
    alphas = [row["alpha"] for row in table["rows"]]
    accuracy = [row["accuracy_pct"] for row in table["rows"]]
    auc = [row["roc_auc_pct"] for row in table["rows"]]

    ax.plot(alphas, accuracy, marker="o", linewidth=2, color="#0f766e", label="Accuracy (%)")
    ax.plot(alphas, auc, marker="s", linewidth=2, color="#2563eb", linestyle="--", label="ROC-AUC (%)")
    best = table["best_alpha"]
    ax.axvline(
        best,
        color="#dc2626",
        linestyle=":",
        linewidth=2,
        label=f"chosen visual weight ({best:.1f} / {1 - best:.1f})",
    )
    ax.scatter([best], [max(accuracy)], color="#dc2626", zorder=5)
    ax.set_xlabel("Visual fusion weight α  (audio weight = 1 − α)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Fusion-weight ablation on the validation set", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", fontsize=8)
    return _to_png_bytes(fig)


_FIGURE_BUILDERS = {
    "confusion_matrices": confusion_matrix_figure,
    "roc_curve": roc_figure,
    "pr_curve": pr_figure,
    "weight_ablation": ablation_figure,
}


def figure_png(name: str) -> Optional[bytes]:
    """Return a cached PNG for a named figure (``None`` if unavailable)."""
    builder = _FIGURE_BUILDERS.get(name)
    if builder is None:
        return None
    with _CACHE_LOCK:
        if name not in _CACHE:
            try:
                _CACHE[name] = builder()
            except Exception:  # noqa: BLE001 - never fail the API on a figure
                _CACHE[name] = None
        return _CACHE[name]


def figures_payload() -> Dict[str, Dict[str, str]]:
    """All figures as ``{name: {"png_base64": ...}}`` for JSON transport."""
    payload: Dict[str, Dict[str, str]] = {}
    for name in _FIGURE_BUILDERS:
        png = figure_png(name)
        if png:
            payload[name] = {"png_base64": base64.b64encode(png).decode("ascii")}
    return payload


def figures_png() -> Dict[str, bytes]:
    """All figures as raw PNG bytes (used by the PDF report)."""
    result: Dict[str, bytes] = {}
    for name in _FIGURE_BUILDERS:
        png = figure_png(name)
        if png:
            result[name] = png
    return result


def clear_cache() -> None:
    """Drop cached figures (tests / settings changes)."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__: List[str] = [
    "ablation_figure",
    "clear_cache",
    "confusion_matrix_figure",
    "figure_png",
    "figures_payload",
    "figures_png",
    "pr_figure",
    "roc_figure",
]
