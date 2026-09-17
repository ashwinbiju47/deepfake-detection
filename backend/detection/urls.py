"""URL routes owned by the detection app (mounted under ``/api/``)."""

from django.urls import path

from . import views

app_name = "detection"

urlpatterns = [
    path("health", views.health, name="health"),
    path("analyses", views.create_analysis, name="create-analysis"),
    path("analyses/ground-truth", views.list_ground_truth, name="list-ground-truth"),
    path("analyses/<uuid:session_id>", views.get_analysis, name="get-analysis"),
    path(
        "analyses/<uuid:session_id>/ground_truth",
        views.set_session_ground_truth,
        name="set-ground-truth",
    ),
    path("analyses/<uuid:session_id>/report", views.get_report, name="get-report"),
    path("evaluations/benchmark", views.get_benchmark, name="get-benchmark"),
    path("evaluations/<uuid:run_id>", views.get_evaluation, name="get-evaluation"),
]


