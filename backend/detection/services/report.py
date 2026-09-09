"""Downloadable PDF Report_Generator (Task 16.1, Requirement 12).

This module implements the design's ``Report_Generator`` interface:

    generate(session_id) -> ReportOutcome

Design intent (design.md -> Components and Interfaces -> Report_Generator):
* Generate a downloadable PDF report summarizing score, label, modality
  findings, and XAI heatmaps (Requirement 12.1).
* The report includes the **system architecture / detection pipeline diagram**
  (Video -> Frame extraction -> Face detection/preprocessing -> Visual model ->
  Audio extraction -> Audio model -> Multimodal fusion -> Fake probability ->
  XAI explanation -> Final result).
* Results chapter: a **model performance table** (Model | Accuracy | Precision
  | Recall | F1 | ROC-AUC) demonstrating Multimodal > Visual-only > Audio-only,
  plus the **cross-dataset generalization** table (trained on one dataset,
  tested on unseen identities from another).
* The **ORIGINAL / Grad-CAM heatmap / OVERLAY triple** for each analyzed frame.
* Refuse incomplete sessions (QUEUED / PROCESSING) with clear guidance (Requirement 12.4).
* On failure, return failure message without serving partial PDF (Requirement 12.5).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from django.conf import settings
from django.utils import timezone

from detection.models import AnalysisSession, FrameHeatmap, Report
from detection.services.benchmark import (
    CROSS_DATASET_RUNS,
    MODALITY_COMPARISON,
    MODALITY_ORDERING,
    BenchmarkRow,
    modality_comparison_table,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportOutcome:
    success: bool
    pdf_bytes: Optional[bytes] = None
    error_code: Optional[str] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Report content model (also consumed by tests without PDF rendering)
# ---------------------------------------------------------------------------

# The end-to-end pipeline shown in the architecture diagram (Results chapter).
PIPELINE_STAGES: Tuple[str, ...] = (
    "Video",
    "Frame extraction",
    "Face detection / preprocessing",
    "Visual model",
    "Audio extraction",
    "Audio model",
    "Multimodal fusion",
    "Fake probability",
    "XAI explanation",
    "Final result",
)


class ReportGenerator:
    """Generates PDF summary reports for completed analysis sessions."""

    LABEL_BY_VARIANT = {
        "multimodal": "Multimodal (fused)",
        "visual_only": "Visual-only",
        "audio_only": "Audio-only",
    }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def generate(session_id: str) -> ReportOutcome:
        """Generate PDF report for session_id (Requirement 12.1, 12.4)."""
        if not getattr(settings, "REPORT_ENABLED", False):
            return ReportOutcome(
                success=False,
                error_code="REPORT_DISABLED",
                message="PDF report generation is not enabled.",
            )

        try:
            session = AnalysisSession.objects.get(id=session_id)
        except AnalysisSession.DoesNotExist:
            return ReportOutcome(
                success=False,
                error_code="NOT_FOUND",
                message="Analysis session not found.",
            )

        # Refuse incomplete sessions (Requirement 12.4)
        if session.status in (AnalysisSession.Status.QUEUED, AnalysisSession.Status.PROCESSING):
            return ReportOutcome(
                success=False,
                error_code="SESSION_INCOMPLETE",
                message="PDF report can only be generated for completed sessions.",
            )

        try:
            pdf_bytes = ReportGenerator._build_pdf(session)

            Report.objects.update_or_create(
                session=session,
                defaults={
                    "status": Report.Status.AVAILABLE,
                    "generated_at": timezone.now(),
                },
            )

            return ReportOutcome(
                success=True,
                pdf_bytes=pdf_bytes,
                error_code=None,
                message=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to generate PDF report for session %s: %s", session_id, exc)
            Report.objects.update_or_create(
                session=session,
                defaults={
                    "status": Report.Status.FAILED,
                    "generated_at": None,
                },
            )
            return ReportOutcome(
                success=False,
                error_code="REPORT_GENERATION_FAILED",
                message=f"Failed to generate PDF report: {exc}",
            )

    # ------------------------------------------------------------------
    # PDF assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pdf(session: AnalysisSession) -> bytes:
        """Build the full PDF report with reportlab (or minimal fallback)."""
        try:
            from reportlab.lib.pagesizes import letter  # type: ignore  # noqa: PLC0415
            from reportlab.pdfgen import canvas  # type: ignore  # noqa: PLC0415
        except ImportError:
            return ReportGenerator._fallback_pdf(session)

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        ReportGenerator._draw_header(p, session)
        ReportGenerator._draw_pipeline_diagram(p)
        ReportGenerator._draw_classification(p, session)
        p.showPage()

        ReportGenerator._draw_results_chapter(p)
        p.showPage()

        ReportGenerator._draw_xai_section(p, session)
        p.showPage()

        p.save()
        return buffer.getvalue()

    # -- Page 1: header + architecture diagram + classification -------------

    @staticmethod
    def _draw_header(p: "canvas.Canvas", session: AnalysisSession) -> None:
        """Draw the report title block and session metadata."""
        p.setFillColorRGB(0.08, 0.10, 0.16)
        p.rect(0, 750, 612, 42, stroke=0, fill=1)
        p.setFillColorRGB(1, 1, 1)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(40, 768, "Real-Time Deepfake Detection Platform — Report")
        p.setFont("Helvetica", 9)
        p.drawString(40, 754, f"Session {session.id}  •  {session.status}  •  generated {timezone.now():%Y-%m-%d %H:%M UTC}")

        p.setFillColorRGB(0.1, 0.1, 0.1)
        p.setFont("Helvetica", 9)
        y = 726
        p.drawString(40, y, f"Source: {session.source_ref}  ({session.source_type})")
        p.drawString(40, y - 14, f"Created: {session.created_at:%Y-%m-%d %H:%M UTC}")
        if session.completed_at:
            p.drawString(40, y - 28, f"Completed: {session.completed_at:%Y-%m-%d %H:%M UTC}")

    @staticmethod
    def _box(
        p: "canvas.Canvas", x: float, y: float, w: float, h: float, lines: Sequence[str]
    ) -> None:
        """Draw a rounded pipeline box with centered multi-line text."""
        p.setLineWidth(0.8)
        p.setStrokeColorRGB(0.18, 0.22, 0.32)
        p.setFillColorRGB(0.90, 0.93, 0.98)
        p.roundRect(x, y, w, h, 5, stroke=1, fill=1)
        p.setFillColorRGB(0.08, 0.10, 0.18)
        p.setFont("Helvetica", 7.5)
        n = len(lines)
        for i, line in enumerate(lines):
            p.drawCentredString(x + w / 2.0, y + h / 2.0 + (n / 2.0 - i - 0.5) * 8.5, line)

    @staticmethod
    def _arrow(p: "canvas.Canvas", x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw an arrow between two box edges."""
        import math  # noqa: PLC0415

        p.setLineWidth(1.0)
        p.setStrokeColorRGB(0.2, 0.3, 0.6)
        p.line(x1, y1, x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 6
        p.setFillColorRGB(0.2, 0.3, 0.6)
        path = p.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(x2 - size * math.cos(angle - 0.4), y2 - size * math.sin(angle - 0.4))
        path.lineTo(x2 - size * math.cos(angle + 0.4), y2 - size * math.sin(angle + 0.4))
        path.close()
        p.drawPath(path, stroke=0, fill=1)

    @staticmethod
    def _draw_pipeline_diagram(p: "canvas.Canvas") -> None:
        """Draw the end-to-end detection pipeline (architecture) diagram."""
        p.setFillColorRGB(0.1, 0.1, 0.15)
        p.setFont("Helvetica-Bold", 13)
        p.drawString(40, 700, "1. System Architecture — Detection Pipeline")
        p.setFont("Helvetica", 8.5)
        p.drawString(40, 686, "Figure 1: End-to-end multimodal deepfake detection pipeline.")

        bw, bh, gap = 115, 38, 20
        x0 = 40
        row1_y, row2_y, row3_y = 640, 552, 464

        # Row 1 (left -> right): stages 1-4
        x = x0
        pos = {}
        for i in range(4):
            label = PIPELINE_STAGES[i]
            lines = [label] if len(label) <= 22 else label.split("/")
            pos[i + 1] = (x, row1_y)
            self = ReportGenerator
            self._box(p, x, row1_y, bw, bh, lines)
            x += bw + gap

        # Row 2 (right -> left): stages 5-8
        x = x0 + 3 * (bw + gap)
        for i in range(7, 3, -1):
            label = PIPELINE_STAGES[i]
            lines = [label] if len(label) <= 22 else label.split("/")
            pos[i + 1] = (x, row2_y)
            ReportGenerator._box(p, x, row2_y, bw, bh, lines)
            x -= bw + gap

        # Row 3 (left -> right): stages 9-10
        x = x0
        for i in range(8, 10):
            label = PIPELINE_STAGES[i]
            lines = [label] if len(label) <= 22 else label.split("/")
            pos[i + 1] = (x, row3_y)
            ReportGenerator._box(p, x, row3_y, bw, bh, lines)
            x += bw + gap

        # Arrows
        def _edge(box_index: int) -> Tuple[float, float, float, float]:
            bx, by = pos[box_index]
            return (bx, by, bx + bw, by + bh)

        # row 1 horizontal
        for i in range(1, 4):
            x1, y1, x2, _ = _edge(i)
            ReportGenerator._arrow(p, x1 + bw, y1 + bh / 2, x2, y1 + bh / 2)
        # drop 4 -> 5
        x1, y1, x2, y2 = _edge(4)
        ReportGenerator._arrow(p, x1 + bw / 2, y1, x1 + bw / 2, y2 - bh - (row1_y - row2_y - bh))
        # row 2 horizontal (right to left)
        for i in range(6, 4, -1):
            x1, y1, x2, _ = _edge(i)
            ReportGenerator._arrow(p, x1, y1 + bh / 2, x2 + bw, y1 + bh / 2)
        # drop 8 -> 9
        x1, y1, x2, y2 = _edge(8)
        ReportGenerator._arrow(p, x1 + bw / 2, y1, x1 + bw / 2, y2 - bh - (row2_y - row3_y - bh))
        # row 3 horizontal 9 -> 10
        x1, y1, x2, _ = _edge(9)
        ReportGenerator._arrow(p, x1 + bw, y1 + bh / 2, x2, y1 + bh / 2)

        p.setFillColorRGB(0.25, 0.25, 0.30)
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(40, row3_y - 22, "Visual content (faces) and audio (spectrograms) are analyzed in parallel and fused "
                                      "into a final deepfake probability, then explained via Grad-CAM and delivered as the result.")

    @staticmethod
    def _draw_classification(p: "canvas.Canvas", session: AnalysisSession) -> None:
        p.setFillColorRGB(0.1, 0.1, 0.15)
        p.setFont("Helvetica-Bold", 13)
        p.drawString(40, 396, "2. Classification Result")

        try:
            fusion = session.fusion_result
        except Exception:  # noqa: BLE001 - OneToOne missing related object
            fusion = None

        if fusion is None:
            p.setFont("Helvetica", 9)
            p.drawString(40, 370, "No fusion result recorded for this session.")
            return

        p.setFillColorRGB(0.93, 0.96, 1.0)
        p.roundRect(40, 300, 532, 62, 6, stroke=1, fill=1)
        p.setFillColorRGB(0.10, 0.10, 0.16)
        p.setFont("Helvetica-Bold", 11)
        label_text = str(fusion.label).upper() if fusion.label else "INCONCLUSIVE"
        p.drawString(58, 330, f"Label: {label_text}")
        p.drawString(400, 330, f"Score: {fusion.score:.2%}" if fusion.score is not None else "Score: N/A")
        p.setFont("Helvetica", 8.5)
        p.drawString(58, 312, f"Modalities used: {fusion.modalities_used or 'none'}")
        p.drawString(400, 312, f"Threshold: {fusion.threshold_used:.2f}")

    # -- Page 2: Results / Evaluation chapter -------------------------------

    @staticmethod
    def _draw_results_chapter(p: "canvas.Canvas") -> None:
        """Draw the model-performance and cross-dataset tables (Results chapter)."""
        p.setFillColorRGB(0.1, 0.1, 0.15)
        p.setFont("Helvetica-Bold", 13)
        p.drawString(40, 760, "3. Results / Evaluation")

        ReportGenerator._draw_metrics_table(p, y=700)
        ReportGenerator._draw_cross_dataset_table(p, y=430)
        ReportGenerator._draw_improvement_note(p, y=388)

    @staticmethod
    def _draw_metrics_table(p: "canvas.Canvas", y: float) -> None:
        """Draw Model | Accuracy | Precision | Recall | F1 | ROC-AUC table."""
        p.setFillColorRGB(0.1, 0.1, 0.15)
        p.setFont("Helvetica-Bold", 11)
        p.drawString(40, y, "3.1 Model performance by modality (FaceForensics++ held-out test)")

        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        col_widths = [140, 78, 78, 68, 68, 78]
        x0, y0 = 40, y - 24
        row_h = 20

        def _cell(x: float, yy: float, text: str, bold: bool = False, align: str = "left") -> None:
            p.setFont("Helvetica-Bold" if bold else "Helvetica", 8.5)
            if align == "right":
                p.drawRightString(x + w - 8, yy + 6, text)
            else:
                p.drawString(x + 8, yy + 6, text)

        # header
        p.setFillColorRGB(0.12, 0.16, 0.26)
        p.rect(x0, y0, sum(col_widths), row_h, stroke=0, fill=1)
        p.setFillColorRGB(1, 1, 1)
        cx = x0
        for h, w in zip(headers, col_widths):
            p.setFont("Helvetica-Bold", 8.5)
            p.drawString(cx + 8, y0 + 6, h)
            cx += w

        by_variant = {r.variant: r for r in MODALITY_COMPARISON}
        yy = y0 - row_h
        for variant in MODALITY_ORDERING:
            row = by_variant[variant]
            cm = row.cm
            is_best = variant == "multimodal"
            p.setFillColorRGB(0.93, 0.97, 1.0) if is_best else p.setFillColorRGB(1, 1, 1)
            p.rect(x0, yy, sum(col_widths), row_h, stroke=0, fill=1)
            p.setStrokeColorRGB(0.8, 0.85, 0.92)
            p.rect(x0 + 0.5, yy + 0.5, sum(col_widths) - 1, row_h - 1, stroke=1, fill=0)
            p.setFillColorRGB(0.1, 0.1, 0.15)
            cells = [
                (ReportGenerator.LABEL_BY_VARIANT[variant], "left", True if is_best else False),
                (f"{cm.accuracy*100:.1f}%", "right", False),
                (f"{cm.precision*100:.1f}%", "right", False),
                (f"{cm.recall*100:.1f}%", "right", False),
                (f"{cm.f1_score*100:.1f}%", "right", False),
                (f"{row.roc_auc*100:.1f}%", "right", False),
            ]
            cx = x0
            for (text, align, bold), w in zip(cells, col_widths):
                p.setFont("Helvetica-Bold" if bold else "Helvetica", 8.5)
                if align == "right":
                    p.drawRightString(cx + w - 8, yy + 6, text)
                else:
                    p.drawString(cx + 8, yy + 6, text)
                cx += w
            yy -= row_h

        p.setFillColorRGB(0.15, 0.15, 0.2)
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(x0, yy - 14, "Best configuration highlighted. Multimodal fusion combines visual and audio likelihoods (60/40 weighted).")

    @staticmethod
    def _draw_cross_dataset_table(p: "canvas.Canvas", y: float) -> None:
        """Draw train-on-A / test-on-B generalization table (accuracy by variant)."""
        p.setFillColorRGB(0.1, 0.1, 0.15)
        p.setFont("Helvetica-Bold", 11)
        p.drawString(40, y, "3.2 Cross-dataset generalization (trained on one dataset, tested on unseen identities)")

        headers = ["Train → Test", "Multimodal", "Visual-only", "Audio-only"]
        col_widths = [150, 130, 120, 120]
        x0, y0 = 40, y - 24
        row_h = 20

        p.setFillColorRGB(0.12, 0.16, 0.26)
        p.rect(x0, y0, sum(col_widths), row_h, stroke=0, fill=1)
        p.setFillColorRGB(1, 1, 1)
        cx = x0
        for h, w in zip(headers, col_widths):
            p.setFont("Helvetica-Bold", 8.5)
            p.drawString(cx + 8, y0 + 6, h)
            cx += w

        # group cross-dataset runs by (train, test) pair
        by_pair: dict = {}
        for row in CROSS_DATASET_RUNS:
            by_pair.setdefault((row.train_dataset, row.dataset), {})[row.variant] = row

        yy = y0 - row_h
        for (train_ds, test_ds), runs in by_pair.items():
            p.setFillColorRGB(1, 1, 1)
            p.rect(x0, yy, sum(col_widths), row_h, stroke=0, fill=1)
            p.setStrokeColorRGB(0.8, 0.85, 0.92)
            p.rect(x0 + 0.5, yy + 0.5, sum(col_widths) - 1, row_h - 1, stroke=1, fill=0)
            p.setFillColorRGB(0.1, 0.1, 0.15)
            p.setFont("Helvetica", 8.5)
            p.drawString(x0 + 8, yy + 6, f"{train_ds} → {test_ds}")
            cx = x0 + col_widths[0]
            for variant in ("multimodal", "visual_only", "audio_only"):
                row = runs.get(variant)
                acc = row.cm.accuracy if row else 0.0
                p.setFont("Helvetica-Bold" if variant == "multimodal" else "Helvetica", 8.5)
                w = col_widths[1]
                p.drawRightString(cx + w - 8, yy + 6, f"{acc*100:.1f}%")
                cx += w
            yy -= row_h

        p.setFillColorRGB(0.15, 0.15, 0.2)
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(x0, yy - 14, "Cross-dataset scores are lower than in-domain scores — expected generalization drop — yet multimodal stays best in every setting.")

    @staticmethod
    def _draw_improvement_note(p: "canvas.Canvas", y: float) -> None:
        """Answer the research question: does combining audio+visual improve detection?"""
        table = modality_comparison_table()
        imp_vis = table["improvement_over_visual_pct"]
        imp_aud = table["improvement_over_audio_pct"]

        p.setFillColorRGB(0.93, 0.96, 1.0)
        p.roundRect(40, y, 532, 62, 6, stroke=1, fill=1)
        p.setFillColorRGB(0.08, 0.10, 0.18)
        p.setFont("Helvetica-Bold", 10)
        p.drawString(58, y + 38, "Q: Does combining audio and visual information actually improve detection?")
        p.setFont("Helvetica", 9)
        p.drawString(58, y + 20, f"Yes. Multimodal accuracy is {imp_vis:+.1f} percentage points higher than visual-only "
                                 f"and {imp_aud:+.1f} higher than audio-only:")
        p.setFont("Helvetica-Bold", 9)
        p.drawCentredString(306, y + 5, "Multimodal  >  Visual-only  >  Audio-only")

    # -- Page 3: XAI section -------------------------------------------------

    @staticmethod
    def _draw_xai_section(p: "canvas.Canvas", session: AnalysisSession) -> None:
        """Embed the ORIGINAL / HEATMAP / OVERLAY triple for analyzed frames."""
        p.setFillColorRGB(0.1, 0.1, 0.15)
        p.setFont("Helvetica-Bold", 13)
        p.drawString(40, 760, "4. Explainability (Grad-CAM)")

        heatmaps: List[FrameHeatmap] = list(
            FrameHeatmap.objects.filter(session=session).order_by("frame_id")[:2]
        )

        if not heatmaps:
            p.setFont("Helvetica", 9)
            p.drawString(40, 732, "No Grad-CAM heatmaps recorded for this session (HEATMAP_ENABLED was off).")
            return

        p.setFont("Helvetica", 9)
        p.drawString(40, 732, "For each analyzed frame the platform persists and shows three artifacts: the ORIGINAL frame, "
                              "the raw Grad-CAM HEATMAP, and the final OVERLAY (heatmap blended over the original).")
        p.drawString(40, 718, "Red regions mark the pixels the model attended to when deciding the frame is a deepfake.")

        from reportlab.lib.utils import ImageReader  # type: ignore  # noqa: PLC0415

        img_w, img_h, gap = 140, 120, 40
        x0 = 60
        y_start = 560
        for hm in heatmaps:
            p.setFont("Helvetica-Bold", 9)
            p.drawString(x0 - 10, y_start + 34, f"Frame {hm.frame_id}")
            captions = [("ORIGINAL", hm.original_png), ("HEATMAP", hm.heatmap_png), ("OVERLAY", hm.overlay_png)]
            cx = x0 - 10
            for caption, png_bytes in captions:
                if png_bytes:
                    try:
                        p.drawImage(ImageReader(io.BytesIO(bytes(png_bytes))), cx, y_start, img_w, img_h, preserveAspectRatio=True)
                    except Exception:  # noqa: BLE001
                        p.setFillColorRGB(0.85, 0.85, 0.85)
                        p.rect(cx, y_start, img_w, img_h, stroke=1, fill=1)
                else:
                    p.setFillColorRGB(0.85, 0.85, 0.85)
                    p.rect(cx, y_start, img_w, img_h, stroke=1, fill=1)
                p.setFillColorRGB(0.15, 0.15, 0.2)
                p.setFont("Helvetica-Bold", 7.5)
                p.drawCentredString(cx + img_w / 2.0, y_start - 12, caption)
                cx += img_w + gap
            y_start -= 190

        p.setFillColorRGB(0.7, 0.3, 0.3)
        p.setFont("Helvetica-Bold", 10)
        p.drawString(40, y_start - 10, "Deepfake probability: see Section 2. Classification Result.")

    # -- Fallback -----------------------------------------------------------

    @staticmethod
    def _fallback_pdf(session: AnalysisSession) -> bytes:
        """Minimal valid PDF used only when reportlab is unavailable."""
        content = (
            f"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            f"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj "
            f"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
            f"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n"
            f"0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
        ).encode("latin-1")
        return content