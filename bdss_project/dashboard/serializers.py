"""Serializers for the prediction API.

The response is designed to be as usable by a machine as the dashboard is by a
person: it carries the probability, the drivers, the recommendation, the owner
and the priority, and it states the model version that produced all of them. An
integration that stores only the probability is throwing away the part of the
answer this framework exists to supply.
"""

from __future__ import annotations

from rest_framework import serializers


class DriverSerializer(serializers.Serializer):
    """One Shapley driver, in the language of the business."""

    feature = serializers.CharField()
    label = serializers.CharField()
    category = serializers.CharField()
    category_label = serializers.CharField()
    direction = serializers.CharField(help_text="'increases' or 'decreases' churn risk")
    contribution = serializers.FloatField(help_text="Signed Shapley value.")
    protected = serializers.BooleanField(
        help_text="A demographic attribute: explained, but never acted on."
    )


class RecommendationSerializer(serializers.Serializer):
    intervention = serializers.CharField()
    owner = serializers.CharField()
    rationale = serializers.CharField()
    rule_id = serializers.CharField()
    priority = serializers.CharField()
    priority_label = serializers.CharField()
    priority_score = serializers.FloatField()
    caveats = serializers.ListField(child=serializers.CharField())
    taxonomy_version = serializers.CharField()


class PredictionResultSerializer(serializers.Serializer):
    customer_ref = serializers.CharField()
    predicted_label = serializers.CharField()
    predicted_churn = serializers.BooleanField()
    probability = serializers.FloatField(help_text="Calibrated probability of churn.")
    confidence = serializers.FloatField(
        help_text="Probability of the class actually assigned. Not the same as the churn probability."
    )
    band = serializers.CharField()
    band_label = serializers.CharField()
    priority = serializers.CharField()
    drivers = DriverSerializer(many=True)
    recommendation = RecommendationSerializer()


class PredictRequestSerializer(serializers.Serializer):
    """The request body for ``POST /api/predict/``.

    ``records`` accepts either a single customer object or a list of them, because
    both are natural things for a caller to send and refusing one of them for
    tidiness is a way of making an API annoying for no benefit.
    """

    model = serializers.CharField(
        required=False,
        help_text=(
            "Model identifier, 'model_key|dataset' (for example "
            "'xgboost|bank_churn'). Defaults to the best registered model by F1."
        ),
    )
    records = serializers.JSONField(
        help_text="A customer object, or a list of them. Keys are feature column names."
    )
    reference_column = serializers.CharField(
        required=False,
        help_text="Column to echo back as the customer reference (e.g. 'CustomerId').",
    )
    top_drivers = serializers.IntegerField(required=False, min_value=1, max_value=10, default=3)

    def validate_records(self, value):
        records = [value] if isinstance(value, dict) else value
        if not isinstance(records, list) or not records:
            raise serializers.ValidationError(
                "Send a customer object, or a non-empty list of them."
            )
        if not all(isinstance(record, dict) for record in records):
            raise serializers.ValidationError("Every record must be a JSON object.")
        if len(records) > 1000:
            raise serializers.ValidationError(
                "The API scores up to 1,000 customers per call. Use the dashboard "
                "upload for larger batches."
            )
        return records
