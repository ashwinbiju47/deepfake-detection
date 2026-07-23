"""API views for the detection app.

Provides the health endpoint and the file-upload intake endpoint
(``POST /api/analyses``, Requirement 1). The upload view is a thin adapter: it
delegates all validation and session creation to
:class:`detection.services.upload.UploadService` and maps the typed
``UploadResult`` onto HTTP responses (``202`` on acceptance, an appropriate
``4xx`` on rejection).

URL submission (Requirement 10) is gated behind ``EXTERNAL_URL_ENABLED`` and
implemented in a later task; this view leaves a clear extension point for it.
"""

from __future__ import annotations

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response

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
                "external_url_enabled": settings.EXTERNAL_URL_ENABLED,
                "heatmap_enabled": settings.HEATMAP_ENABLED,
                "report_enabled": settings.REPORT_ENABLED,
            },
            "worker_concurrency": settings.CELERY_WORKER_CONCURRENCY,
        }
    )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def create_analysis(request: Request) -> Response:
    """Submit a video for analysis (``POST /api/analyses``).

    Multipart file uploads are validated and accepted by the ``UploadService``.
    A successful submission returns ``202 Accepted`` with the new
    ``session_id``; a rejected submission returns the matching ``4xx`` status
    with the typed ``error_code`` and message.

    JSON URL submissions are recognized but gated behind
    ``EXTERNAL_URL_ENABLED`` (Requirement 10, Task 14) — they currently return
    ``501 Not Implemented`` as a clear extension point.
    """
    upload = request.FILES.get("file")
    if upload is not None:
        result = UploadService().submit_file(upload)
        if result["accepted"]:
            return Response(
                {"session_id": result["session_id"]},
                status=status.HTTP_202_ACCEPTED,
            )
        return Response(
            {"error_code": result["error_code"], "message": result["message"]},
            status=_ERROR_STATUS.get(
                result["error_code"], status.HTTP_400_BAD_REQUEST
            ),
        )

    # --- URL submission (Requirement 10, Task 14) ------------------------
    if "url" in request.data:
        url_str = str(request.data.get("url") or "")
        result = UploadService().submit_url(url_str)
        if result["accepted"]:
            return Response(
                {"session_id": result["session_id"]},
                status=status.HTTP_202_ACCEPTED,
            )
        error_code = result["error_code"] or "BAD_REQUEST"
        status_code = status.HTTP_400_BAD_REQUEST
        if error_code == "BAD_SCHEME" or error_code == "URL_INPUT_DISABLED":
            status_code = status.HTTP_400_BAD_REQUEST
        elif error_code == "TOO_LARGE":
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        elif error_code == "URL_UNREACHABLE" or error_code == "UNDECODABLE":
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

        return Response(
            {"error_code": error_code, "message": result["message"]},
            status=status_code,
        )


    return Response(
        {
            "error_code": "NO_INPUT",
            "message": "No file was provided. Submit a multipart 'file' field.",
        },
        status=status.HTTP_400_BAD_REQUEST,
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
                "split": eval_run.split,
                "accuracy": eval_run.accuracy,
                "meets_baseline": eval_run.meets_baseline,
                "timestamp": eval_run.timestamp.isoformat() if eval_run.timestamp else None,
                "metrics": {
                    "confusion_matrix": metrics.confusion_matrix if metrics else {},
                    "precision": metrics.precision if metrics else 0.0,
                    "recall": metrics.recall if metrics else 0.0,
                    "f1_score": metrics.f1_score if metrics else 0.0,
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


