"""The audit trail.

Every score the system serves is written down here, together with the drivers
that explained it, the recommendation those drivers produced, and the version of
the model that made it. This is not logging for its own sake. When a retention
decision is later questioned -- by a manager, an auditor, or the customer
concerned -- the framework has to be able to reproduce the prediction, the
explanation and the recommendation exactly as they stood at the time, and a score
without its model version cannot do that.

The override fields are the other half of the same commitment. The system is
advisory: a retention manager can reject any recommendation, and when they do,
the rejection is recorded with its reason. That log serves two purposes. It feeds
the next model review, and it is the only way to detect automation bias -- the
tendency to defer to the system precisely because it is a system. A queue with no
overrides is not a sign of a good model.
"""

from __future__ import annotations

from django.conf import settings
from django.db import models
from django.urls import reverse

SOURCE_CHOICES = [("web", "Dashboard upload"), ("api", "REST API")]

BAND_CHOICES = [("high", "High risk"), ("medium", "Medium risk"), ("low", "Low risk")]

PRIORITY_CHOICES = [
    ("P1", "P1 - act now"),
    ("P2", "P2 - act this cycle"),
    ("P3", "P3 - monitor"),
]

OVERRIDE_CHOICES = [
    ("accepted", "Accepted as recommended"),
    ("different_offer", "Actioned, but with a different offer"),
    ("no_action", "No action taken - recommendation rejected"),
    ("wrong_risk", "Risk estimate looks wrong"),
    ("already_resolved", "Already handled or recently renewed"),
]


class PredictionBatch(models.Model):
    """One scored customer list: the unit a campaign is planned from."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="batches",
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    source_filename = models.CharField(max_length=255)
    model_key = models.CharField(max_length=64)
    model_label = models.CharField(max_length=128)
    model_version = models.CharField(max_length=32)
    dataset = models.CharField(max_length=64)
    threshold = models.FloatField(null=True, blank=True)
    taxonomy_version = models.CharField(max_length=32, blank=True)

    row_count = models.PositiveIntegerField(default=0)
    flagged_count = models.PositiveIntegerField(default=0)
    summary = models.JSONField(default=dict, blank=True)
    #: Columns the model expected but the upload did not carry. Kept because a
    #: batch scored against a half-empty schema is a batch to distrust.
    missing_columns = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "prediction batches"

    def __str__(self) -> str:
        return f"{self.source_filename} ({self.row_count} rows, {self.model_label})"

    def get_absolute_url(self) -> str:
        return reverse("dashboard:results", args=[self.pk])

    @property
    def flagged_pct(self) -> float:
        return round(100 * self.flagged_count / self.row_count, 1) if self.row_count else 0.0


class Prediction(models.Model):
    """One customer, one score, one explanation, one recommendation."""

    batch = models.ForeignKey(
        PredictionBatch,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="predictions",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="predictions",
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    source = models.CharField(max_length=8, choices=SOURCE_CHOICES, default="web")

    # --- what was scored -----------------------------------------------------
    customer_ref = models.CharField(max_length=128, blank=True)
    input_data = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "The customer's feature values as supplied. Retention of this field is "
            "governed by BDSS_LOG_INPUT_DATA; switching it off keeps the reasons and "
            "the score while discarding the personal data behind them."
        ),
    )

    # --- what scored it ------------------------------------------------------
    model_key = models.CharField(max_length=64, db_index=True)
    model_label = models.CharField(max_length=128)
    model_version = models.CharField(max_length=32, db_index=True)
    dataset = models.CharField(max_length=64)
    threshold = models.FloatField(null=True, blank=True)

    # --- what it said --------------------------------------------------------
    predicted_label = models.CharField(max_length=64)
    predicted_churn = models.BooleanField(default=False)
    probability = models.FloatField(help_text="Calibrated probability of churn.")
    confidence = models.FloatField(
        help_text="Probability of the class the model actually assigned."
    )
    band = models.CharField(max_length=8, choices=BAND_CHOICES)

    # --- why it said it ------------------------------------------------------
    top_drivers = models.JSONField(
        default=list, help_text="Ranked SHAP drivers, strongest first."
    )

    # --- what to do about it -------------------------------------------------
    recommendation = models.CharField(max_length=255)
    business_owner = models.CharField(max_length=64)
    priority = models.CharField(max_length=4, choices=PRIORITY_CHOICES, db_index=True)
    priority_score = models.FloatField(default=0.0)
    rule_id = models.CharField(max_length=64, blank=True)
    # The rationale is copied rather than looked up. The rule table is editable by
    # design, so the reason a customer was routed somewhere in March must be
    # readable in September even if the rule has since been reworded.
    rationale = models.TextField(blank=True)
    taxonomy_version = models.CharField(max_length=32, blank=True)
    caveats = models.JSONField(default=list, blank=True)

    # --- what the human did about it ----------------------------------------
    override_action = models.CharField(
        max_length=32, choices=OVERRIDE_CHOICES, blank=True
    )
    override_reason = models.TextField(blank=True)
    overridden_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="overrides",
    )
    overridden_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["batch", "priority"]),
            models.Index(fields=["created_at", "model_key"]),
        ]

    def __str__(self) -> str:
        return f"{self.customer_ref or self.pk}: {self.probability:.0%} ({self.model_label})"

    @property
    def is_overridden(self) -> bool:
        return bool(self.override_action)

    @property
    def probability_pct(self) -> float:
        return round(100 * self.probability, 1)

    @property
    def band_label(self) -> str:
        return dict(BAND_CHOICES).get(self.band, self.band)
