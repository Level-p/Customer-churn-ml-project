"""Forms for the upload and override flows.

The model selector is built from the registry at request time rather than from a
hard-coded list, so a model trained this morning is selectable this afternoon
without touching the code, and a model that has been withdrawn cannot be chosen
at all.
"""

from __future__ import annotations

from django import forms
from django.conf import settings

from dashboard.models import OVERRIDE_CHOICES
from ml_engine import predictor


class UploadForm(forms.Form):
    """Upload a customer list and choose the model that will score it."""

    csv_file = forms.FileField(
        label="Customer list (CSV)",
        help_text=(
            "One row per customer, with a header row. Columns the model does not "
            "recognise are ignored, so an export straight from the CRM is fine."
        ),
        widget=forms.ClearableFileInput(attrs={"accept": ".csv,text/csv"}),
    )
    model_id = forms.ChoiceField(
        label="Prediction model",
        help_text="Every model is served from the registry with its version recorded.",
    )
    only_at_risk = forms.BooleanField(
        label="Show only customers above the operating threshold",
        required=False,
        initial=False,
        help_text=(
            "The full scored list is exported either way; this only filters the "
            "table on screen."
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        models = predictor.available_models()
        self.fields["model_id"].choices = [
            (
                model["id"],
                "{label} - F1 {f1} | trained {version}".format(
                    label=model["label"],
                    f1=_format_metric(model["metrics"].get("f1")),
                    version=model["version"][:8],
                ),
            )
            for model in models
        ]
        if models:
            # Default to the strongest F1 among the registered models, but the
            # choice stays the user's: the framework recommends, it does not lock.
            best = max(models, key=lambda model: model["metrics"].get("f1") or 0)
            self.fields["model_id"].initial = best["id"]
        else:
            self.fields["model_id"].help_text = (
                "No models are registered. Run `python -m ml.train_all` first."
            )

    def clean_csv_file(self):
        uploaded = self.cleaned_data["csv_file"]
        if not uploaded.name.lower().endswith(".csv"):
            raise forms.ValidationError("The file must be a .csv export.")
        if uploaded.size > settings.BDSS_MAX_UPLOAD_BYTES:
            limit = settings.BDSS_MAX_UPLOAD_BYTES / (1024 * 1024)
            raise forms.ValidationError(
                f"The file is larger than the {limit:.0f} MB limit. Split it into "
                "smaller batches."
            )
        return uploaded


def _format_metric(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


class OverrideForm(forms.Form):
    """Record a human decision that departs from the recommendation.

    The reason is mandatory. An override with no reason tells the next model
    review nothing, and telling the next model review something is the entire
    purpose of capturing it.
    """

    action = forms.ChoiceField(
        label="What did you do?",
        choices=OVERRIDE_CHOICES,
        widget=forms.RadioSelect,
    )
    reason = forms.CharField(
        label="Why?",
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "e.g. customer renewed last week"}),
        max_length=1000,
    )
