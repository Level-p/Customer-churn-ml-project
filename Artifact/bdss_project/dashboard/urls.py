"""URLs for the dashboard app."""

from __future__ import annotations

from django.urls import path

from dashboard import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("batch/<int:pk>/", views.results, name="results"),
    path("batch/<int:pk>/export/", views.export, name="export"),
    path("prediction/<int:pk>/override/", views.override, name="override"),
    path("history/", views.history, name="history"),
    path("taxonomy/", views.taxonomy, name="taxonomy"),
    path("registry/", views.registry_view, name="registry"),
    path("model/<str:model_key>/<str:dataset>/", views.model_card, name="model_card"),
]
