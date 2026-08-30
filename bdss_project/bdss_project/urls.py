"""Root URL configuration.

    /               the dashboard (login required)
    /accounts/      login and logout
    /api/predict/   JSON prediction endpoint
    /api/models/    the model registry, as served
    /admin/         Django admin, for the prediction and override logs
"""

from __future__ import annotations

from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

from dashboard import api

urlpatterns = [
    path("", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
    path(
        "accounts/login/",
        auth_views.LoginView.as_view(
            template_name="registration/login.html", redirect_authenticated_user=True
        ),
        name="login",
    ),
    path(
        "accounts/logout/",
        auth_views.LogoutView.as_view(),  # POST only in Django 4.2+, by design
        name="logout",
    ),
    path("api/predict/", api.PredictView.as_view(), name="api-predict"),
    path("api/models/", api.ModelListView.as_view(), name="api-models"),
    path("admin/", admin.site.urls),
]
