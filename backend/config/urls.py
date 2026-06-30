"""Root URL configuration.

API routes live under ``/api/`` and are owned by the ``detection`` app.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("detection.urls")),
]
