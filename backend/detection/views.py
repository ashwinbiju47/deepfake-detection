"""API views for the detection app.

Provides the health endpoint, the file-upload intake endpoint
(``POST /api/analyses``, Requirement 1), the session-status endpoint used by
the dashboard to show the metrics of the last analysis, and the evaluation
endpoints. The upload view is a thin adapter: it delegates all validation and
session creation to :class:`detection.services.upload.UploadService` and maps
the typed ``UploadResult`` onto HTTP responses (``202`` on acceptance, an
appropriate ``4xx`` on rejection).

Only direct file uploads (video / image / audio) are accepted; URL intake has
been removed by product decision.
"""

from __future__ import annotations

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response

from detection.models import AnalysisSession
from detection.services.ablation import ablation_table
from detection.services.benchmark import (
    cross_dataset_table,
    curves_payload,
    modality_comparison_table,
)
from detection.services.figures import figures_payload
from detection.services.upload import (
    ERROR_EMPTY_FILE,
    ERROR_TOO_LARGE,
    ERROR_UNDECODABLE,
    ERROR_UNSUPPORTED_FORMAT,
    UploadService,
)

# Map each typed validation error code to an appropriate 4xx status code.
_ERROR_STATUS = {
    ERROR_EMPTY_FILE: status.HTTP_400_BAD_REQUEST,
    ERROR_UNSUPPORTED_FORMAT: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ERROR_TOO_LARGE: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    ERROR_UNDECODABLE: status.HTTP_422_UNPROCESSABLE_ENTITY,
}


@api_view(["GET"])
def health(_request: Request) -> Response:
    """Report service liveness and the active feature-flag configuration."""
    return Response(
        {
            "status": "ok",
            "feature_flags": {
                "heatmap_enabled": settings.HEATMAP_ENABLED,
                "report_enabled": settings.REPORT_ENABLED,
            },
            "worker_concurrency": settings.CELERY_WORKER_CONCURRENCY,
        }
    )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def create_analysis(request: Request) -> Response:
    """Submit a media file for analysis (``POST /api/analyses``).

    Accepts **video, image, or audio** files (multipart). URL submission has
    been removed: a request carrying a ``url`` field instead of a file is
    rejected with ``URL_NOT_SUPPORTED`` so clients get a clear signal rather
    than a generic 400.

    A successful submission returns ``202 Accepted`` with the new
    ``session_id``; a rejected submission returns the matching ``4xx`` status
    with the typed ``error_code`` and message.
    """
    if "url" in request.data and request.FILES.get("file") is None:
        return Response(
            {
                "error_code": "URL_NOT_SUPPORTED",
                "message": "URL submission is not supported. Upload a video, image, or audio file.",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    upload = request.FILES.get("file")
    if upload is not None:
        result = UploadService().submit_file(upload)
        if result["accepted"]:
            return Response(
                {
                    "session_id": result["session_id"],
                    "media_kind": result.get("media_kind"),
                },
                status=status.HTTP_202_ACCEPTED,
            )
        return Response(
            {"error_code": result["error_code"], "message": result["message"]},
            status=_ERROR_STATUS.get(
                result["error_code"], status.HTTP_400_BAD_REQUEST
            ),
        )

    return Response(
        {
            "error_code": "NO_INPUT",
            "message": "No file was provided. Submit a multipart 'file' field.",
        },
        status=status.HTTP_400_BAD_REQUEST,
    )


@api_view(["GET"])
def get_analysis(_request: Request, session_id: str) -> Response:
    """Fetch the metrics of one analysis session (``GET /api/analyses/{id}``).

    Returns the persisted per-modality evidence and the fused classification
    for the session, so the dashboard can show the metrics **of the last
    analysis** rather than constant reference numbers.
    """
    try:
        session = AnalysisSession.objects.get(id=session_id)
    except (AnalysisSession.DoesNotExist, ValueError):
        return Response(
            {"error_code": "NOT_FOUND", "message": "Analysis session not found."},
            status=status.HTTP_404_NOT_FOUND,
        )

    visual = getattr(session, "visual_result", None)
    audio = getattr(session, "audio_result", None)
    fusion = getattr(session, "fusion_result", None)

    return Response(
        {
            "id": str(session.id),
            "status": session.status,
            "media_kind": session.media_kind,
            "source_ref": session.source_ref,
            "media_state": session.media_state,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "visual": {
                "likelihood": visual.aggregate_likelihood if visual else None,
                "state": visual.state if visual else None,
                "frames_analyzed": visual.frames_analyzed if visual else 0,
                "faces_isolated": visual.faces_isolated if visual else 0,
                "error_detail": visual.error_detail if visual else "",
            },
            "audio": {
                "likelihood": audio.likelihood if audio else None,
                "state": audio.state if audio else None,
                "error_detail": audio.error_detail if audio else "",
            },
            "fusion": (
                {
                    "score": fusion.score,
                    "label": fusion.label,
                    "modalities_used": (
                        fusion.modalities_used.split(",") if fusion.modalities_used else []
                    ),
                    "inconclusive": fusion.inconclusive,
                    "threshold_used": fusion.threshold_used,
                }
                if fusion
                else None
            ),
        }
    )


@api_view(["GET"])
def get_evaluation(_request: Request, run_id: str) -> Response:
    """Fetch model evaluation details (``GET /api/evaluations/<run_id>``, Requirement 8.3)."""
    from detection.models import ModelEvaluation

    try:
        eval_run = ModelEvaluation.objects.get(run_id=run_id)
        metrics = getattr(eval_run, "metrics", None)
        return Response(
            {
                "run_id": str(eval_run.run_id),
                "dataset": eval_run.dataset,
                "train_dataset": eval_run.train_dataset,
                "variant": eval_run.variant,
                "split": eval_run.split,
                "accuracy": eval_run.accuracy,
                "meets_baseline": eval_run.meets_baseline,
                "evaluated_at": (
                    eval_run.evaluated_at.isoformat() if eval_run.evaluated_at else None
                ),
                "metrics": {
                    "confusion_matrix": metrics.confusion_matrix if metrics else {},
                    "precision": metrics.precision if metrics else 0.0,
                    "recall": metrics.recall if metrics else 0.0,
                    "f1_score": metrics.f1_score if metrics else 0.0,
                    "roc_auc": metrics.roc_auc if metrics else None,
                }
                if metrics
                else None,
            }
        )
    except ModelEvaluation.DoesNotExist:
        return Response(
            {"error_code": "NOT_FOUND", "message": "Evaluation run not found."},
            status=status.HTTP_404_NOT_FOUND,
        )


@api_view(["GET"])
def get_benchmark(_request: Request) -> Response:
    """Return the Results-chapter benchmark tables (``GET /api/evaluations/benchmark``).

    Serves the modality-comparison table (Multimodal > Visual-only >
    Audio-only, with per-model Accuracy / Precision / Recall / F1 / ROC-AUC),
    the cross-dataset generalization table (train on one dataset, test on
    another with unseen identities), the evaluation figures (confusion
    matrices, ROC, precision-recall, fusion-weight ablation) and the ROC/PR
    curve data behind them.
    """
    payload = {
        "modality_comparison": modality_comparison_table(),
        "cross_dataset": cross_dataset_table(),
        "curves": curves_payload(),
        "weight_ablation": ablation_table(
            configured_alpha=settings.FUSION_WEIGHT_VISUAL
        ),
        "figures": figures_payload(),
    }
    return Response(payload)


@api_view(["GET"])
def get_report(_request: Request, session_id: str) -> Response:
    """Generate and serve PDF report (``GET /api/analyses/<session_id>/report``, Requirement 12.2)."""
    from django.http import HttpResponse
    from detection.services.report import ReportGenerator

    result = ReportGenerator.generate(session_id)
    if result.success and result.pdf_bytes:
        response = HttpResponse(result.pdf_bytes, content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="report_{session_id}.pdf"'
        return response

    error_code = result.error_code or "FAILED"
    status_code = status.HTTP_400_BAD_REQUEST
    if error_code == "SESSION_INCOMPLETE" or error_code == "REPORT_DISABLED":
        status_code = status.HTTP_400_BAD_REQUEST
    elif error_code == "NOT_FOUND":
        status_code = status.HTTP_404_NOT_FOUND
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return Response(
        {"error_code": error_code, "message": result.message},
        status=status_code,
    )


