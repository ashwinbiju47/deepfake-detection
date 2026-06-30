"""URL routes owned by the detection app (mounted under ``/api/``)."""

from django.urls import path

from . import views

app_name = "detection"

urlpatterns = [
    path("health", views.health, name="health"),
    path("analyses", views.create_analysis, name="create-analysis"),
]
