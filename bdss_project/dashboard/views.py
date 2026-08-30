"""Views for the retention decision-support dashboard.

The interface assumes a user who knows the customers and the offers, not the
models. Nothing on any screen requires statistical literacy: risk arrives as a
band, drivers arrive in the language of the business, and every row ends in a
recommended action with an owner. The numbers are all still there, one click
away, for the user who wants them.

Every screen carries the model version and the scoring date, so that any decision
taken from it can be traced back to the model that informed it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Count, Q
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone as django_timezone
from django.views.decorators.http import require_POST

from dashboard.forms import OverrideForm, UploadForm
from dashboard.models import OVERRIDE_CHOICES, Prediction, PredictionBatch
from ml import decision, registry
from ml.config import DATASETS
from ml_engine import predictor, utils

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #


@login_required
def index(request):
    """What the system has done lately, and what it is currently able to do."""
    batches = PredictionBatch.objects.select_related("user")[:8]
    predictions = Prediction.objects.all()

    overrides = predictions.exclude(override_action="").count()
    total = predictions.count()
    priority_counts = dict(
        predictions.values_list("priority")
        .annotate(count=Count("id"))
        .values_list("priority", "count")
    )

    context = {
        "models": predictor.available_models(),
        "batches": batches,
        "totals": {
            "predictions": total,
            "batches": PredictionBatch.objects.count(),
            "flagged": predictions.filter(predicted_churn=True).count(),
            "overrides": overrides,
            # An override rate of zero is not a triumph. It is the signature of
            # automation bias: nobody is disagreeing with the machine.
            "override_rate": round(100 * overrides / total, 1) if total else 0.0,
            "priorities": priority_counts,
        },
        "recent": predictions.select_related("batch")[:10],
    }
    return render(request, "dashboard/index.html", context)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@login_required
def upload(request):
    """Take a customer list, score it, and write the batch to the audit trail."""
    if request.method != "POST":
        return render(
            request,
            "dashboard/upload.html",
            {"form": UploadForm(), "models": predictor.available_models()},
        )

    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(
            request,
            "dashboard/upload.html",
            {"form": form, "models": predictor.available_models()},
        )

    uploaded = form.cleaned_data["csv_file"]
    model_id = form.cleaned_data["model_id"]

    try:
        frame = utils.read_uploaded_csv(uploaded, max_rows=settings.BDSS_MAX_UPLOAD_ROWS)
    except utils.UploadError as exc:
        form.add_error("csv_file", str(exc))
        return render(
            request,
            "dashboard/upload.html",
            {"form": form, "models": predictor.available_models()},
        )

    reference_column = utils.guess_reference_column(frame)
    try:
        results, meta = predictor.predict(
            frame,
            identifier=model_id,
            top_drivers=settings.BDSS_TOP_DRIVERS,
            reference_column=reference_column,
        )
    except predictor.ModelUnavailable as exc:
        form.add_error("model_id", str(exc))
        return render(
            request,
            "dashboard/upload.html",
            {"form": form, "models": predictor.available_models()},
        )
    except Exception as exc:  # noqa: BLE001
        # A failure here is almost always a schema mismatch: the right model, the
        # wrong file. Say so, rather than showing a stack trace to a retention
        # manager who has no way to act on it.
        LOGGER.exception("Scoring failed for %s with %s", uploaded.name, model_id)
        form.add_error(
            None,
            "The file could not be scored with this model. The most likely cause is "
            "that the customer list belongs to a different domain than the model was "
            f"trained on. ({exc})",
        )
        return render(
            request,
            "dashboard/upload.html",
            {"form": form, "models": predictor.available_models()},
        )

    summary = utils.summarise(results)
    batch = _persist(request, uploaded.name, results, meta, summary)

    messages.success(
        request,
        f"Scored {len(results):,} customers with {meta['model_label']} "
        f"(version {meta['model_version']}). "
        f"{summary['flagged']:,} are above the operating threshold.",
    )
    if meta["missing_columns"]:
        messages.warning(
            request,
            "The upload was missing "
            f"{len(meta['missing_columns'])} column(s) the model expects: "
            f"{', '.join(meta['missing_columns'][:6])}"
            f"{'...' if len(meta['missing_columns']) > 6 else ''}. "
            "They were imputed, so treat these scores with caution.",
        )

    destination = f"{batch.get_absolute_url()}?only_at_risk=1" if form.cleaned_data[
        "only_at_risk"
    ] else batch.get_absolute_url()
    return redirect(destination)


@transaction.atomic
def _persist(request, filename: str, results: list[dict], meta: dict, summary: dict) -> PredictionBatch:
    """Write the batch and its predictions in one transaction.

    A batch whose rows were half-written would be an audit trail that lies, so
    either all of it lands or none of it does.
    """
    batch = PredictionBatch.objects.create(
        user=request.user,
        source_filename=filename[:255],
        model_key=meta["model_key"],
        model_label=meta["model_label"],
        model_version=meta["model_version"],
        dataset=meta["dataset"],
        threshold=meta["threshold"],
        taxonomy_version=str(meta.get("taxonomy_version") or ""),
        row_count=len(results),
        flagged_count=summary.get("flagged", 0),
        summary=summary,
        missing_columns=meta.get("missing_columns", []),
    )

    keep_inputs = settings.BDSS_LOG_INPUT_DATA
    Prediction.objects.bulk_create(
        [
            Prediction(
                batch=batch,
                user=request.user,
                source="web",
                customer_ref=str(result["customer_ref"])[:128],
                input_data=result["input_data"] if keep_inputs else {},
                model_key=meta["model_key"],
                model_label=meta["model_label"],
                model_version=meta["model_version"],
                dataset=meta["dataset"],
                threshold=meta["threshold"],
                predicted_label=str(result["predicted_label"]),
                predicted_churn=result["predicted_churn"],
                probability=result["probability"],
                confidence=result["confidence"],
                band=result["band"],
                top_drivers=result["drivers"],
                recommendation=result["recommendation"]["intervention"][:255],
                business_owner=result["recommendation"]["owner"][:64],
                priority=result["priority"],
                priority_score=result["priority_score"],
                rule_id=result["recommendation"]["rule_id"][:64],
                rationale=result["recommendation"]["rationale"],
                taxonomy_version=str(meta.get("taxonomy_version") or ""),
                caveats=result["recommendation"]["caveats"],
            )
            for result in results
        ],
        batch_size=1000,
    )
    return batch


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@login_required
def results(request, pk: int):
    """The work queue for one batch: sorted, explained, and actionable."""
    batch = get_object_or_404(PredictionBatch.objects.select_related("user"), pk=pk)
    only_at_risk = request.GET.get("only_at_risk") == "1"

    queryset = batch.predictions.all()
    if only_at_risk:
        queryset = queryset.filter(predicted_churn=True)

    # The queue order *is* the product: P1 first, then by the priority score,
    # which folds the customer's value into their risk.
    queryset = queryset.order_by("priority", "-priority_score", "-probability")

    paginator = Paginator(queryset, settings.BDSS_RESULTS_PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))

    probabilities = [
        {"probability": value}
        for value in batch.predictions.values_list("probability", flat=True)
    ]
    charts = {
        "risk": utils.risk_distribution_chart(probabilities, batch.threshold),
        "priority": utils.priority_chart(batch.summary),
        "drivers": utils.driver_category_chart(batch.summary),
    }

    context = {
        "batch": batch,
        "page": page,
        "charts": charts,
        "summary": batch.summary,
        "only_at_risk": only_at_risk,
        # The rows render their own override form: a bound Django form repeated
        # fifty times down a page would emit fifty copies of the same element id.
        "override_choices": OVERRIDE_CHOICES,
        "band_colours": utils.BAND_COLOURS,
        "overridden": batch.predictions.exclude(override_action="").count(),
    }
    return render(request, "dashboard/results.html", context)


@login_required
def export(request, pk: int):
    """Download the scored list, with its reasons and its model version attached."""
    batch = get_object_or_404(PredictionBatch, pk=pk)
    rows = batch.predictions.order_by("priority", "-priority_score")

    results = [
        {
            "customer_ref": row.customer_ref,
            "probability": row.probability,
            "confidence": row.confidence,
            "predicted_label": row.predicted_label,
            "band_label": row.band_label,
            "priority": row.priority,
            "priority_score": row.priority_score,
            "drivers": row.top_drivers,
            "recommendation": {
                "intervention": row.recommendation,
                "owner": row.business_owner,
                "rationale": row.rationale,
                "caveats": row.caveats,
            },
        }
        for row in rows
    ]
    meta = {
        "model_label": batch.model_label,
        "model_version": batch.model_version,
        "scored_at": batch.created_at.isoformat(),
    }

    response = HttpResponse(utils.results_to_csv(results, meta), content_type="text/csv")
    stamp = batch.created_at.strftime("%Y%m%d")
    response["Content-Disposition"] = (
        f'attachment; filename="retention_queue_{batch.pk}_{stamp}.csv"'
    )
    return response


@login_required
@require_POST
def override(request, pk: int):
    """Record that a human disagreed, and why. The system proposes; staff dispose."""
    prediction = get_object_or_404(Prediction, pk=pk)
    form = OverrideForm(request.POST)

    if not form.is_valid():
        messages.error(request, "An override needs both an action and a reason.")
    else:
        prediction.override_action = form.cleaned_data["action"]
        prediction.override_reason = form.cleaned_data["reason"]
        prediction.overridden_by = request.user
        prediction.overridden_at = django_timezone.now()
        prediction.save(
            update_fields=[
                "override_action", "override_reason", "overridden_by", "overridden_at"
            ]
        )
        messages.success(
            request,
            f"Override recorded for {prediction.customer_ref or prediction.pk}. "
            "It will be reviewed with the next model refresh.",
        )

    if prediction.batch_id:
        return redirect(f"{prediction.batch.get_absolute_url()}#prediction-{prediction.pk}")
    return redirect("dashboard:history")


@login_required
def history(request):
    """Every batch this system has scored, newest first."""
    paginator = Paginator(PredictionBatch.objects.select_related("user"), 25)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "dashboard/history.html", {"page": page})


# --------------------------------------------------------------------------- #
# Accountability
# --------------------------------------------------------------------------- #


@login_required
def model_card(request, model_key: str, dataset: str):
    """The registered model, in full: how it scores, what drives it, whom it fails.

    This screen exists because a model that cannot be interrogated cannot be
    trusted with a retention budget. It shows the evaluation on the untouched test
    partition, the leading drivers, the result of auditing SHAP against LIME, and
    the fairness audit -- including the disparities the model has not solved.
    """
    try:
        loaded = predictor.load_model(model_key, dataset)
    except predictor.ModelUnavailable as exc:
        raise Http404(str(exc)) from exc

    bundle = loaded.bundle
    evaluation = bundle.get("evaluation", {})
    fairness = bundle.get("fairness", {})

    # The dataset description is read from the live configuration rather than from
    # the artefact. What we know about a dataset can change after a model is
    # trained on it -- a leak found, a label discovered to be synthetic -- and that
    # discovery has to reach the person reading the metrics, not sit frozen inside
    # a pickle written before anyone knew.
    dataset_config = DATASETS.get(bundle["dataset"])
    description = (
        dataset_config.description if dataset_config else bundle.get("dataset_description", "")
    )

    context = {
        "bundle": bundle,
        "dataset_description": description,
        "metrics": bundle.get("metrics", {}),
        "evaluation": evaluation,
        "confusion": _confusion_rows(evaluation),
        "audit": bundle.get("explanation_audit", {}),
        "fairness": fairness,
        "importance_chart": utils.global_importance_chart(bundle.get("global_importance", [])),
        "importance": bundle.get("global_importance", [])[:12],
        "hyperparameters": bundle.get("hyperparameters", {}),
        "calibration": bundle.get("calibration", {}),
        "snapshot": bundle.get("training_snapshot", {}),
        "threshold_choice": bundle.get("threshold_choice") or {},
        "cross_validation": bundle.get("cross_validation", {}),
        "profit": evaluation.get("profit") or {},
        "usage": Prediction.objects.filter(
            model_key=model_key, model_version=bundle["version"]
        ).aggregate(
            served=Count("id"), overridden=Count("id", filter=~Q(override_action=""))
        ),
    }
    return render(request, "dashboard/model_card.html", context)


def _confusion_rows(evaluation: dict) -> list[dict]:
    """Lay the confusion matrix out for a table, with the class labels attached."""
    matrix = evaluation.get("confusion_matrix") or []
    classes = evaluation.get("classes") or []
    rows = []
    for index, row in enumerate(matrix):
        rows.append(
            {
                "label": str(classes[index]) if index < len(classes) else str(index),
                "cells": row,
                "total": sum(row),
            }
        )
    return rows


@login_required
def taxonomy(request):
    """Publish the intervention taxonomy.

    Publishing it is a fairness safeguard as much as a usability one. If offers
    flow only to the customers the model flags, the groups it never flags quietly
    receive worse treatment, and a rule table nobody outside the team can read
    makes that invisible. It is also the file a retention manager edits when the
    offers change, so they are entitled to see what it currently says.
    """
    try:
        table = decision.load_rules()
    except decision.RuleTableError as exc:
        raise Http404(str(exc)) from exc

    usage = dict(
        Prediction.objects.values_list("rule_id")
        .annotate(count=Count("id"))
        .values_list("rule_id", "count")
    )
    rows = decision.taxonomy_table(table)
    for row in rows:
        row["times_applied"] = usage.get(row["id"], 0)

    return render(
        request,
        "dashboard/taxonomy.html",
        {
            "table": table,
            "rows": rows,
            "path": settings.BDSS_RULES_PATH,
            "notes": table.get("notes", []),
            "unmatched": usage.get("default_manual_review", 0),
        },
    )


@login_required
def registry_view(request):
    """The model registry: every version ever trained, and what it scored."""
    entries = registry.list_entries()
    for entry in entries:
        entry["served"] = Prediction.objects.filter(
            model_key=entry["model_key"], model_version=entry["version"]
        ).count()
    return render(
        request,
        "dashboard/registry.html",
        {"entries": entries, "now": datetime.now(timezone.utc)},
    )
