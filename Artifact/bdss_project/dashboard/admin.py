"""Admin views over the audit trail.

Predictions are read-only here. The log is evidence: it records what the system
told a member of staff at a moment in time, and evidence that can be edited after
the fact is not evidence. Overrides are made through the dashboard, where they
are attributed and reasoned; the admin exists to search and inspect, not to
rewrite.
"""

from __future__ import annotations

from django.contrib import admin
from django.utils.html import format_html

from dashboard.models import Prediction, PredictionBatch


@admin.register(PredictionBatch)
class PredictionBatchAdmin(admin.ModelAdmin):
    list_display = (
        "id", "created_at", "source_filename", "model_label", "model_version",
        "row_count", "flagged_count", "user",
    )
    list_filter = ("model_key", "dataset", "created_at")
    search_fields = ("source_filename", "model_version")
    readonly_fields = tuple(field.name for field in PredictionBatch._meta.fields)
    date_hierarchy = "created_at"

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id", "created_at", "customer_ref", "risk", "priority", "recommendation",
        "business_owner", "model_label", "model_version", "override_action",
    )
    list_filter = (
        "priority", "band", "source", "model_key", "model_version",
        "override_action", "created_at",
    )
    search_fields = ("customer_ref", "recommendation", "business_owner", "rule_id")
    date_hierarchy = "created_at"
    readonly_fields = tuple(field.name for field in Prediction._meta.fields)

    @admin.display(description="Churn risk", ordering="probability")
    def risk(self, obj: Prediction) -> str:
        colour = {"high": "#b42318", "medium": "#b54708", "low": "#067647"}[obj.band]
        # Colour plus the number and the band name: the colour never carries the
        # meaning on its own.
        return format_html(
            '<span style="color:{};font-weight:600">{}%</span> {}',
            colour, obj.probability_pct, obj.band_label,
        )

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False
