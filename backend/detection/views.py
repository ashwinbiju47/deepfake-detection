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

    # --- URL submission extension point (Requirement 10, Task 14) ---------
    if "url" in request.data:
        if not settings.EXTERNAL_URL_ENABLED:
            return Response(
                {
                    "error_code": "URL_INPUT_DISABLED",
                    "message": "External URL input is not enabled.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        # Implemented in Task 14 (scheme/reachability/streaming-size guard).
        return Response(
            {
                "error_code": "NOT_IMPLEMENTED",
                "message": "URL submission is not yet implemented.",
            },
            status=status.HTTP_501_NOT_IMPLEMENTED,
        )

    return Response(
        {
            "error_code": "NO_INPUT",
            "message": "No file was provided. Submit a multipart 'file' field.",
        },
        status=status.HTTP_400_BAD_REQUEST,
    )
