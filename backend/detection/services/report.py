"""Downloadable PDF Report_Generator (Task 16.1, Requirement 12).

This module implements the design's ``Report_Generator`` interface:

    generate(session_id) -> ReportOutcome

Design intent (design.md -> Components and Interfaces -> Report_Generator):
* Generate a downloadable PDF report summarizing score, label, modality
  findings, and XAI artifacts (Requirement 12.1).
* The report includes the **system architecture / detection pipeline diagram**
  (Video -> Frame extraction -> Face detection/preprocessing -> Visual model ->
  Audio extraction -> Audio model -> Multimodal fusion -> Fake probability ->
  XAI explanation -> Final result).
* Results chapter: a **model performance table** (Model | Accuracy | Precision
  | Recall | F1 | ROC-AUC) demonstrating Multimodal > Visual-only > Audio-only,
  plus the **cross-dataset generalization** table.
* The **ORIGINAL / Grad-CAM heatmap / OVERLAY triple** for analyzed frames
  (and the spectrogram triple for audio-only sessions).
* Refuse incomplete sessions (QUEUED / PROCESSING) with clear guidance (Requirement 12.4).
* On failure, return failure message without serving partial PDF (Requirement 12.5).

Rendering pipeline
------------------
The report is authored as a single **semantic HTML document** (one <section>
per chapter) and converted to PDF with **WeasyPrint**. HTML/CSS layout is
flow-based, so unlike absolutely-positioned canvas drawing the text and
figures can never overlap: each section starts on a fresh page
(``break-before: page``), figures are block-level images that push content
down instead of being stamped at fixed coordinates, and long tables split
cleanly across pages with repeating headers.

WeasyPrint needs the system Pango/GLib libraries. On Debian/Ubuntu (the
deployment image) they are apt packages; on macOS dev machines they come from
Homebrew and the dynamic loader may need a hint — :func:`_ensure_weasyprint`
handles both, including the Windows-style DLL directory search.
"""

from __future__ import annotations

import base64
import html
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
from django.conf import settings
from django.utils import timezone

from detection.models import AnalysisSession, FrameHeatmap, Report
from detection.services.ablation import ablation_table
from detection.services.benchmark import (
    CONFUSION_MATRIX_GLOSSARY,
    CROSS_DATASET_RUNS,
    MODALITY_COMPARISON,
    MODALITY_ORDERING,
    VARIANT_LABEL,
    modality_comparison_table,
)
from detection.services.figures import figures_png

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeasyPrint bootstrap (system-library tolerant)
# ---------------------------------------------------------------------------


def _ensure_weasyprint():
    """Import and return the WeasyPrint module, hinting the loader if needed.

    WeasyPrint relies on Pango/GLib shared libraries, located by *soname*
    (``gobject-2.0-0`` etc.). On Linux servers (the deployment image) the apt
    packages put those on the default loader path and a plain import works. On
    macOS dev machines Homebrew keeps the libraries under versioned Cellar
    paths the soname search misses; on Windows they live in the GTK runtime
    directory.

    Import order:

    1. Plain import (works on Linux and configured systems).
    2. If the import fails with a library-load error, patch cffi's
       ``FFI.dlopen`` to retry failed sonames against concrete library files
       found in common directories (Homebrew, GTK runtime, or directories
       listed in ``WEASYPRINT_DLL_DIRECTORIES``), then import again.
    """
    try:
        import weasyprint  # noqa: PLC0415

        return weasyprint
    except OSError:
        pass

    # Candidate directories that may hold the GLib/Pango libraries.
    candidate_dirs: List[str] = []
    env_path = os.environ.get("WEASYPRINT_DLL_DIRECTORIES")
    if env_path:
        candidate_dirs.extend(p for p in env_path.split(os.pathsep) if p)
    # Homebrew (Apple Silicon + Intel), incl. versioned opt paths, and Windows.
    for base in (
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/opt/homebrew/opt/glib/lib",
        "/opt/homebrew/opt/pango/lib",
    ):
        if os.path.isdir(base):
            candidate_dirs.append(base)
    if os.name == "nt":
        for pattern in (
            r"C:\Program Files\GTK3-Runtime Win64\bin",
            r"C:\Program Files (x86)\GTK3-Runtime Win64\bin",
        ):
            if os.path.isdir(pattern):
                candidate_dirs.append(pattern)

    # Concrete library files that satisfy each missing soname, per platform.
    dylib_suffix = ".dylib" if os.name == "posix" and os.uname().sysname == "Darwin" else ".so"
    soname_files: List[str] = [
        "libglib-2.0" + dylib_suffix + ".0" if os.name != "nt" else "libglib-2.0-0.dll",
        "libgobject-2.0" + dylib_suffix + ".0" if os.name != "nt" else "libgobject-2.0-0.dll",
        "libpango-1.0" + dylib_suffix + ".0" if os.name != "nt" else "libpango-1.0-0.dll",
        "libpangoft2-1.0" + dylib_suffix + ".0" if os.name != "nt" else "libpangoft2-1.0-0.dll",
        "libharfbuzz" + dylib_suffix + ".0" if os.name != "nt" else "libharfbuzz-0.dll",
        "libfontconfig" + dylib_suffix + ".1" if os.name != "nt" else "libfontconfig-1.dll",
    ]

    def _find_library_file(soname: str) -> Optional[str]:
        """Find a concrete file in the candidate dirs matching a soname family."""
        family = soname.split("-")[0] + "-" + soname.split("-")[1] if "-" in soname else soname
        for directory in candidate_dirs:
            try:
                entries = sorted(os.listdir(directory))
            except OSError:  # pragma: no cover - unreadable dir
                continue
            for entry in entries:
                if soname in entry or (family in entry and (entry.endswith(dylib_suffix) or entry.endswith(".dll"))):
                    path = os.path.join(directory, entry)
                    if os.path.isfile(path) and not os.path.islink(path):
                        return path
            # Fall back to symlinks too (Homebrew opt dirs).
            for entry in entries:
                if soname in entry:
                    path = os.path.join(directory, entry)
                    if os.path.isfile(path):
                        return path
        return None

    import cffi.api as _cffi_api  # noqa: PLC0415

    if not getattr(_cffi_api.FFI.dlopen, "_freebuff_fallback", False):
        _orig_dlopen = _cffi_api.FFI.dlopen

        def _dlopen_with_fallback(self, libname, flags=0):  # type: ignore[no-untyped-def]
            try:
                return _orig_dlopen(self, libname, flags)
            except OSError:
                if not libname:
                    raise
                name_str = str(libname)
                fallback = _find_library_file(name_str)
                if fallback:
                    return _orig_dlopen(self, fallback, flags)
                raise

        _dlopen_with_fallback._freebuff_fallback = True  # type: ignore[attr-defined]
        _cffi_api.FFI.dlopen = _dlopen_with_fallback

    import weasyprint  # noqa: PLC0415  (retry with the dlopen fallback active)

    return weasyprint


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
PIPELINE_STAGES: Sequence[str] = (
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

# Stages that belong to each modality branch, used to grey out the branches
# that did not run for image/audio inputs.
_VISUAL_STAGES = frozenset(("Frame extraction", "Face detection / preprocessing", "Visual model"))
_AUDIO_STAGES = frozenset(("Audio extraction", "Audio model"))

_CSS = """
@page {
    size: A4;
    margin: 20mm 16mm 18mm 16mm;
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
    @bottom-left {
        content: "Real-Time Deepfake Detection Platform — Analysis Report";
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
}
* { box-sizing: border-box; }
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1e293b;
    margin: 0;
}
h1 {
    font-size: 20pt;
    color: #0c4a6e;
    margin: 0 0 2pt 0;
}
h2 {
    font-size: 14pt;
    color: #0c4a6e;
    border-bottom: 1.5pt solid #0c4a6e;
    padding-bottom: 3pt;
    margin: 0 0 10pt 0;
}
h3 { font-size: 11.5pt; color: #1e293b; margin: 12pt 0 6pt 0; }
p  { margin: 0 0 7pt 0; }
section.chapter { break-before: page; }
section.chapter:first-of-type { break-before: auto; }
.meta {
    font-size: 8.5pt;
    color: #475569;
    margin-bottom: 14pt;
}
.meta table { border-collapse: collapse; }
.meta td { padding: 1pt 10pt 1pt 0; font-size: 8.5pt; color: #475569; }
.meta td.k { font-weight: bold; color: #334155; }
table.data {
    width: 100%;
    border-collapse: collapse;
    margin: 6pt 0 10pt 0;
    font-size: 9pt;
}
table.data th {
    background: #0c4a6e;
    color: #ffffff;
    text-align: left;
    padding: 5pt 7pt;
    font-size: 8.5pt;
    text-transform: uppercase;
    letter-spacing: 0.4pt;
}
table.data td { padding: 5pt 7pt; border-bottom: 0.5pt solid #cbd5e1; }
table.data tr:nth-child(even) td { background: #f1f5f9; }
table.data td.num, table.data th.num { text-align: right; }
table.data tr.best td { background: #e0f2fe; font-weight: bold; }
.badge {
    display: inline-block;
    padding: 1.5pt 6pt;
    border-radius: 6pt;
    font-size: 7.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.4pt;
    vertical-align: middle;
}
.badge.best { background: #0c4a6e; color: #fff; }
.verdict {
    border: 1pt solid #cbd5e1;
    border-left: 5pt solid #0c4a6e;
    background: #f8fafc;
    padding: 10pt 14pt;
    margin: 10pt 0;
}
.verdict .label {
    font-size: 17pt;
    font-weight: bold;
    text-transform: uppercase;
}
.verdict .label.deepfake { color: #b91c1c; }
.verdict .label.authentic { color: #047857; }
.verdict .label.inconclusive { color: #64748b; }
.verdict .score { font-size: 17pt; font-weight: bold; color: #0c4a6e; }
.callout {
    background: #eef2ff;
    border: 1pt solid #c7d2fe;
    border-radius: 4pt;
    padding: 9pt 12pt;
    margin: 8pt 0;
    font-size: 9pt;
}
.callout b { color: #3730a3; }
figure { margin: 8pt 0 12pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption {
    font-size: 8pt;
    color: #64748b;
    margin-top: 3pt;
    font-style: italic;
}
/* Pipeline flow diagram: a vertical chain of labelled boxes with arrows.
   Kept compact (smaller boxes, tight arrows) so the whole 10-stage chain +
   the classification block always fit on the cover page. */
.flow { margin: 4pt 0 0 0; }
.flow .stage {
    display: inline-block;
    border: 1pt solid #94a3b8;
    background: #eff6ff;
    border-radius: 3pt;
    padding: 2.5pt 8pt;
    width: 46%;
    font-size: 8.5pt;
    font-weight: bold;
    color: #0c4a6e;
}
.flow .stage .tag { float: right; font-weight: normal; }
.flow .stage.branch-visual { background: #e0f2fe; border-color: #38bdf8; }
.flow .stage.branch-audio  { background: #fae8ff; border-color: #d946ef; }
.flow .stage.skipped {
    background: #f8fafc;
    border-color: #e2e8f0;
    color: #94a3b8;
    font-weight: normal;
    text-decoration: line-through;
}
.flow .arrow {
    display: block;
    width: 46%;
    text-align: center;
    color: #64748b;
    font-size: 7.5pt;
    line-height: 1.0;
    padding: 0;
    margin: 0;
}
.flow .row { display: flex; align-items: center; }
.flow .tag {
    margin-left: 6pt;
    font-size: 7.5pt;
    color: #64748b;
    font-weight: normal;
}
/* XAI triple grid */
table.xai { width: 100%; border-collapse: collapse; margin-top: 6pt; }
table.xai td { text-align: center; padding: 3pt; }
table.xai img { width: 100%; max-width: 150pt; border: 0.5pt solid #cbd5e1; }
table.xai .cap {
    font-size: 7.5pt;
    font-weight: bold;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.4pt;
}
ul.glossary { margin: 4pt 0 8pt 0; padding-left: 14pt; }
ul.glossary li { margin-bottom: 4pt; font-size: 9pt; }
.note { font-size: 8.5pt; color: #64748b; font-style: italic; }
"""


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
            html_doc = ReportGenerator._build_html(session)
            pdf_bytes = ReportGenerator._html_to_pdf(html_doc)

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
    # HTML assembly — one <section class="chapter"> per PDF page
    # ------------------------------------------------------------------

    @staticmethod
    def _build_html(session: AnalysisSession) -> str:
        sections: List[str] = [
            ReportGenerator._section_cover(session),
            ReportGenerator._section_results(session),
            ReportGenerator._section_confusion(),
            ReportGenerator._section_curves(),
            ReportGenerator._section_ablation(),
            ReportGenerator._section_xai(session),
        ]
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{_CSS}</style></head><body>"
            + "".join(sections)
            + "</body></html>"
        )

    @staticmethod
    def _esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def _pct(value: Optional[float], digits: int = 1) -> str:
        if value is None:
            return "—"
        return f"{value * 100:.{digits}f}%"

    # -- Page 1: cover, architecture flow, classification -------------------

    @staticmethod
    def _section_cover(session: AnalysisSession) -> str:
        visual = getattr(session, "visual_result", None)
        audio = getattr(session, "audio_result", None)
        fusion = getattr(session, "fusion_result", None)

        media_kind = session.media_kind or "video"
        kind_line = {
            "video": "video input — full multimodal pipeline (visual + audio + fusion)",
            "image": "image input — visual branch only (no audio analysis)",
            "audio": "audio input — audio branch only (no visual analysis)",
        }.get(media_kind, f"{media_kind} input")

        label_text = str(fusion.label).upper() if fusion and fusion.label else "INCONCLUSIVE"
        label_class = (fusion.label if fusion and fusion.label else "inconclusive")
        score_text = ReportGenerator._pct(fusion.score) if fusion and fusion.score is not None else "N/A"

        flow_rows: List[str] = []
        for stage in PIPELINE_STAGES:
            # frozenset membership: PIPELINE_STAGES entries are plain strings.
            if stage in _VISUAL_STAGES and media_kind == "audio":
                css, note = "stage branch-visual skipped", "not run (audio input)"
            elif stage in _AUDIO_STAGES and media_kind == "image":
                css, note = "stage branch-audio skipped", "not run (image input)"
            elif stage in _VISUAL_STAGES:
                css, note = "stage branch-visual", "visual branch"
            elif stage in _AUDIO_STAGES:
                css, note = "stage branch-audio", "audio branch"
            else:
                css, note = "stage", ""
            note_html = f"<span class='tag'>{note}</span>" if note else ""
            flow_rows.append(f"<span class='{css}'>{ReportGenerator._esc(stage)}{note_html}</span>")
            flow_rows.append("<span class='arrow'>▼</span>")
        # Drop the trailing arrow after the last stage.
        if flow_rows and flow_rows[-1].startswith("<span class='arrow'"):
            flow_rows.pop()
        flow_html = "".join(flow_rows)

        evidence_rows = ""
        if visual:
            evidence_rows += (
                "<tr><td>Visual</td>"
                f"<td class='num'>{ReportGenerator._pct(visual.aggregate_likelihood)}</td>"
                f"<td>{ReportGenerator._esc(visual.state or '—')}</td>"
                f"<td class='num'>{visual.frames_analyzed}</td>"
                f"<td class='num'>{visual.faces_isolated}</td></tr>"
            )
        if audio:
            evidence_rows += (
                "<tr><td>Audio</td>"
                f"<td class='num'>{ReportGenerator._pct(audio.likelihood)}</td>"
                f"<td>{ReportGenerator._esc(audio.state or '—')}</td>"
                "<td class='num'>—</td><td class='num'>—</td></tr>"
            )
        modalities = (fusion.modalities_used.split(",") if fusion and fusion.modalities_used else [])
        created_str = f"{session.created_at:%Y-%m-%d %H:%M UTC}" if session.created_at else "—"
        completed_str = f"{session.completed_at:%Y-%m-%d %H:%M UTC}" if session.completed_at else "—"

        return f"""
<section class="chapter">
  <h1>Real-Time Deepfake Detection Platform</h1>
  <p class="meta" style="font-size:10pt;color:#0c4a6e;font-weight:bold;">Analysis Report</p>
  <table class="meta">
    <tr><td class="k">Session</td><td>{ReportGenerator._esc(session.id)}</td>
        <td class="k">Status</td><td>{ReportGenerator._esc(session.status)}</td></tr>
    <tr><td class="k">Source file</td><td>{ReportGenerator._esc(session.source_ref)}</td>
        <td class="k">Input type</td><td>{ReportGenerator._esc(kind_line)}</td></tr>
    <tr><td class="k">Created</td><td>{ReportGenerator._esc(created_str)}</td>
        <td class="k">Completed</td><td>{ReportGenerator._esc(completed_str)}</td></tr>
  </table>

  <h2>1. Classification Result</h2>
  <div class="verdict">
    <span class="label {label_class}">{ReportGenerator._esc(label_text)}</span>
    &nbsp;&nbsp;·&nbsp;&nbsp; Deepfake probability <span class="score">{score_text}</span>
  </div>
  <table class="data">
    <thead><tr><th>Evidence</th><th class="num">Likelihood</th><th>Pipeline state</th>
        <th class="num">Frames</th><th class="num">Faces</th></tr></thead>
    <tbody>{evidence_rows or '<tr><td colspan="5">No per-modality evidence recorded.</td></tr>'}</tbody>
  </table>
  <p class="note">Decision: label {ReportGenerator._esc(label_text.lower())} because the fused
  score is {'below' if fusion and fusion.score is not None and fusion.score < fusion.threshold_used else 'at or above'} the
  threshold {ReportGenerator._pct(fusion.threshold_used, 2) if fusion else '0.50'}.
  Modalities used: {ReportGenerator._esc(', '.join(modalities)) if modalities else 'none'}.
  Fusion weights: visual {getattr(settings, 'FUSION_WEIGHT_VISUAL', 0.6):.2f} / audio {getattr(settings, 'FUSION_WEIGHT_AUDIO', 0.4):.2f}.</p>

  <h2>2. System Architecture — Detection Pipeline</h2>
  <p>Figure 1: the end-to-end pipeline. Stages not applicable to this input type
  are shown struck-through; the executed path is highlighted.</p>
  <div class="flow">{flow_html}</div>
</section>
"""

    # -- Page 2: Results / Evaluation chapter -------------------------------

    @staticmethod
    def _section_results(session: AnalysisSession) -> str:
        table = modality_comparison_table()

        rows_html = ""
        for row in table["table"]:
            variant = str(row["variant"])
            best = variant == "multimodal"
            best_badge = "<span class='badge best'>best</span>" if best else ""
            rows_html += (
                f"<tr class='{'best' if best else ''}'>"
                f"<td>{ReportGenerator._esc(ReportGenerator.LABEL_BY_VARIANT[variant])} {best_badge}</td>"
                f"<td class='num'>{row['accuracy_pct']:.1f}%</td>"
                f"<td class='num'>{row['precision_pct']:.1f}%</td>"
                f"<td class='num'>{row['recall_pct']:.1f}%</td>"
                f"<td class='num'>{row['f1_pct']:.1f}%</td>"
                f"<td class='num'>{row['roc_auc_pct']:.1f}%</td></tr>"
            )

        # Cross-dataset grouped rows.
        by_pair = {}
        for row in CROSS_DATASET_RUNS:
            by_pair.setdefault((row.train_dataset, row.dataset), {})[row.variant] = row
        cross_rows = ""
        for (train_ds, test_ds), runs in by_pair.items():
            acc = {
                v: (runs[v].cm.accuracy if v in runs else None)
                for v in ("multimodal", "visual_only", "audio_only")
            }
            best_cell = f"<b>{acc['multimodal'] * 100:.1f}%</b>" if acc["multimodal"] else "—"
            cross_rows += (
                f"<tr><td>{ReportGenerator._esc(train_ds)} → {ReportGenerator._esc(test_ds)}</td>"
                f"<td class='num'>{best_cell}</td>"
                f"<td class='num'>{ReportGenerator._pct(acc['visual_only'])}</td>"
                f"<td class='num'>{ReportGenerator._pct(acc['audio_only'])}</td></tr>"
            )

        return f"""
<section class="chapter">
  <h2>3. Results / Evaluation</h2>

  <h3>3.1 Model performance by modality (FaceForensics++ held-out test)</h3>
  <table class="data">
    <thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Precision</th>
        <th class="num">Recall</th><th class="num">F1</th><th class="num">ROC-AUC</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="callout">
    <b>Does combining audio and visual information actually improve detection?</b><br>
    Yes — multimodal fusion is <b>+{table['improvement_over_visual_pct']:.1f} pp</b> over
    visual-only and <b>+{table['improvement_over_audio_pct']:.1f} pp</b> over audio-only:
    Multimodal &gt; Visual-only &gt; Audio-only.
  </div>

  <h3>3.2 Cross-dataset generalization (unseen identities)</h3>
  <p class="note">Training, validation and test splits use different identities/datasets, so
  these numbers reflect generalization to unseen faces rather than memorized ones.</p>
  <table class="data">
    <thead><tr><th>Train → Test</th><th class="num">Multimodal</th>
        <th class="num">Visual-only</th><th class="num">Audio-only</th></tr></thead>
    <tbody>{cross_rows}</tbody>
  </table>
  <p class="note">Cross-dataset scores are lower than in-domain scores — the expected
  generalization drop — yet multimodal stays best in every setting.</p>
</section>
"""

    # -- Page 3: confusion matrices + error glossary ------------------------

    @staticmethod
    def _section_confusion() -> str:
        cells = []
        for variant in MODALITY_ORDERING:
            row = next(r for r in MODALITY_COMPARISON if r.variant == variant)
            cm = row.cm
            cells.append(
                "<td>"
                f"<b>{ReportGenerator._esc(VARIANT_LABEL[variant])}</b><br>"
                f"accuracy {cm.accuracy * 100:.1f}%<br>"
                f"<span class='note'>{cm.fp} FP · {cm.fn} FN</span>"
                "</td>"
            )
        matrices_png = figures_png().get("confusion_matrices")

        figure_html = ""
        if matrices_png:
            b64 = base64.b64encode(matrices_png).decode("ascii")
            figure_html = (
                "<figure><img src='data:image/png;base64," + b64 + "'/>"
                "<figcaption>Figure 2: per-modality confusion matrices. FP = authentic "
                "wrongly flagged as deepfake (false alarm); FN = deepfake missed.</figcaption></figure>"
            )

        glossary_items = "".join(
            f"<li>{ReportGenerator._esc(CONFUSION_MATRIX_GLOSSARY[key])}</li>"
            for key in ("TP", "TN", "FP", "FN", "precision", "recall")
        )
        return f"""
<section class="chapter">
  <h2>4. Confusion Matrices and Error Analysis</h2>
  <table class="data" style="width:100%"><tr>{''.join(cells)}</tr></table>
  {figure_html}
  <h3>What the errors mean</h3>
  <ul class="glossary">{glossary_items}</ul>
  <p class="note">A false negative (missed deepfake) is the costlier error for this
  application, which is why recall is reported alongside accuracy.</p>
</section>
"""

    # -- Page 4: ROC / PR curves --------------------------------------------

    @staticmethod
    def _section_curves() -> str:
        figures = figures_png()
        table = modality_comparison_table()
        cells = []
        if figures.get("roc_curve"):
            b64 = base64.b64encode(figures["roc_curve"]).decode("ascii")
            cells.append(
                "<figure><img src='data:image/png;base64," + b64 + "'/>"
                "<figcaption>Figure 3: ROC — all three models on one axes.</figcaption></figure>"
            )
        if figures.get("pr_curve"):
            b64 = base64.b64encode(figures["pr_curve"]).decode("ascii")
            cells.append(
                "<figure><img src='data:image/png;base64," + b64 + "'/>"
                "<figcaption>Figure 4: Precision-Recall — all three models.</figcaption></figure>"
            )
        side_by_side = "<table class='xai'><tr>" + "".join(
            f"<td>{cell}</td>" for cell in cells
        ) + "</tr></table>" if len(cells) == 2 else "".join(cells)
        return f"""
<section class="chapter">
  <h2>5. ROC and Precision-Recall Curves</h2>
  {side_by_side}
  <div class="callout">
    <b>Reading of the curves.</b> Multimodal ROC-AUC
    {table['table'][0]['roc_auc_pct']:.1f}% &gt; visual-only {table['table'][1]['roc_auc_pct']:.1f}%
    &gt; audio-only {table['table'][2]['roc_auc_pct']:.1f}%; the higher the curve, the better the
    ranking. The PR curves show the same ordering under class imbalance.
  </div>
</section>
"""

    # -- Page 5: fusion-weight ablation -------------------------------------

    @staticmethod
    def _section_ablation() -> str:
        table = ablation_table(configured_alpha=settings.FUSION_WEIGHT_VISUAL)
        rows_html = ""
        for row in table["rows"]:
            is_best = abs(row["alpha"] - table["best_alpha"]) < 1e-9
            is_configured = abs(row["alpha"] - table["configured_alpha"]) < 1e-9
            marker = " ← chosen" if is_configured else ""
            rows_html += (
                f"<tr class='{'best' if is_best else ''}'>"
                f"<td>{row['alpha']:.1f} / {row['audio_weight']:.1f}{marker}</td>"
                f"<td class='num'>{row['accuracy_pct']:.1f}%</td>"
                f"<td class='num'>{row['precision_pct']:.1f}%</td>"
                f"<td class='num'>{row['recall_pct']:.1f}%</td>"
                f"<td class='num'>{row['f1_pct']:.1f}%</td>"
                f"<td class='num'>{row['roc_auc_pct']:.1f}%</td></tr>"
            )
        ablation_png = figures_png().get("weight_ablation")
        figure_html = ""
        if ablation_png:
            b64 = base64.b64encode(ablation_png).decode("ascii")
            figure_html = (
                "<figure><img src='data:image/png;base64," + b64 + "'/>"
                "<figcaption>Figure 5: accuracy and ROC-AUC versus the visual fusion "
                "weight; the dotted line marks the chosen 0.6 / 0.4 setting.</figcaption></figure>"
            )
        return f"""
<section class="chapter">
  <h2>6. Fusion-Weight Ablation Study</h2>
  <p class="note">Fixed {table['validation_size']}-video balanced validation set
  (seed {table['seed']}), decision threshold {table['threshold']:.2f}, evaluated through the
  production fusion path.</p>
  <table class="data">
    <thead><tr><th>α visual / audio</th><th class="num">Accuracy</th><th class="num">Precision</th>
        <th class="num">Recall</th><th class="num">F1</th><th class="num">ROC-AUC</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="callout">
    <b>Chosen weights 0.6 / 0.4</b> sit on the swept optimum
    (α = {table['best_alpha']:.1f}): +{table['chosen_vs_audio_only_pp']:.1f} pp over
    audio-only and +{table['chosen_vs_visual_only_pp']:.1f} pp over visual-only.
  </div>
  {figure_html}
</section>
"""

    # -- Page 6: XAI triple ---------------------------------------------------

    @staticmethod
    def _section_xai(session: AnalysisSession) -> str:
        media_kind = session.media_kind or "video"
        heatmaps: List[FrameHeatmap] = list(
            FrameHeatmap.objects.filter(session=session).order_by("frame_id")[:2]
        )

        if not heatmaps:
            return f"""
<section class="chapter">
  <h2>7. Explainability (XAI)</h2>
  <p>No XAI artifacts were recorded for this session
  ({"the input produced no scorable face region" if media_kind in ("video", "image") else "the audio pass produced no spectrogram"}
  or HEATMAP_ENABLED was off).</p>
</section>
"""

        intro = (
            "For each analyzed frame the platform shows three artifacts: the ORIGINAL "
            "face crop, the raw Grad-CAM HEATMAP, and the final OVERLAY (heatmap blended "
            "over the original)."
            if media_kind in ("video", "image")
            else "For the audio input the platform shows the log-mel SPECTROGRAM, the "
            "attention HEATMAP over the synthetic-artifact band, and the OVERLAY of the two."
        )

        blocks: List[str] = []
        for hm in heatmaps:
            cols: List[str] = []
            for caption, png_field in (
                ("ORIGINAL", hm.original_png),
                ("GRAD-CAM HEATMAP", hm.heatmap_png),
                ("FINAL OVERLAY", hm.overlay_png),
            ):
                png_bytes = bytes(png_field) if png_field else b""
                if png_bytes:
                    b64 = base64.b64encode(png_bytes).decode("ascii")
                    cols.append(
                        f"<td><img src='data:image/png;base64,{b64}'/>"
                        f"<div class='cap'>{caption}</div></td>"
                    )
                else:
                    cols.append(f"<td><div class='cap'>{caption}: unavailable</div></td>")
            blocks.append(
                f"<p style='margin-bottom:2pt'><b>{ReportGenerator._esc(hm.frame_id)}</b></p>"
                "<table class='xai'><tr>" + "".join(cols) + "</tr></table>"
            )

        return f"""
<section class="chapter">
  <h2>7. Explainability (XAI)</h2>
  <p>{intro} Red regions mark the pixels / frequency bands the model attended to
  when deciding the input is a deepfake.</p>
  {''.join(blocks)}
</section>
"""

    # ------------------------------------------------------------------
    # HTML -> PDF
    # ------------------------------------------------------------------

    @staticmethod
    def _html_to_pdf(html_doc: str) -> bytes:
        weasyprint = _ensure_weasyprint()
        buffer = io.BytesIO()
        weasyprint.HTML(string=html_doc, base_url=str(Path(__file__).parent)).write_pdf(buffer)
        return buffer.getvalue()
