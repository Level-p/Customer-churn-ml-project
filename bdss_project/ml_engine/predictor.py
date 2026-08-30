"""Load registered models and run inference.

This module is the join between the two halves of the system. It reads the
artefacts written by the training pipeline, scores customers with them, explains
each score with the same Shapley machinery the training run audited, and passes
the resulting drivers through the intervention taxonomy to produce a recommended
action with an owner and a priority.

Three properties are load-bearing and are asserted rather than assumed:

* the probability served is the *calibrated* one, mapped through the isotonic
  calibrators fitted on the training partition's out-of-fold predictions;
* the label served is the one the *registered operating threshold* implies, not
  the arg-max at 0.5;
* the model version that produced every score is recorded with that score, so a
  decision can be reproduced after the model has moved on.

Models are cached in-process and reloaded automatically when the registry entry
changes, so retraining does not require restarting the web server.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml import decision, registry
from ml import explain as explanation_layer
from ml.calibration import apply_calibration
from ml.data_prep import CLASSIFIER_STEP, align_columns, transform_features
from ml.evaluation import labels_from_proba

LOGGER = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[str, "LoadedModel"] = {}


class ModelUnavailable(RuntimeError):
    """No model is registered under the requested key.

    Raised rather than silently falling back to another model: a retention
    decision must never be attributed to a model that did not make it.
    """


@dataclass
class LoadedModel:
    """A model artefact, plus the explainer built lazily on top of it."""

    key: str
    bundle: dict[str, Any]
    artifact_path: Path
    loaded_mtime: float
    _explainer: Any = None

    @property
    def model_key(self) -> str:
        return self.bundle["model_key"]

    @property
    def dataset(self) -> str:
        return self.bundle["dataset"]

    @property
    def version(self) -> str:
        return self.bundle["version"]

    def explainer(self):
        """Build the SHAP explainer once, from the artefact's own background sample."""
        if self._explainer is None:
            with _LOCK:
                if self._explainer is None:
                    estimator = self.bundle["pipeline"].named_steps[CLASSIFIER_STEP]
                    self._explainer = explanation_layer.build_explainer(
                        estimator, self.bundle["background"], self.bundle["model_key"]
                    )
        return self._explainer


# --------------------------------------------------------------------------- #
# Registry access
# --------------------------------------------------------------------------- #


def model_id(model_key: str, dataset: str) -> str:
    """A model is identified by what it is *and* what it was trained on.

    A random forest trained on banking data and one trained on streaming data are
    different models with different feature schemas; collapsing them onto the
    label "Random Forest" is how the wrong model ends up scoring the wrong file.
    """
    return f"{model_key}|{dataset}"


def available_models() -> list[dict]:
    """Every model the dashboard can currently serve, newest version of each."""
    entries = registry.list_entries()
    newest: dict[str, dict] = {}
    for entry in sorted(entries, key=lambda item: item["version"]):
        newest[model_id(entry["model_key"], entry["dataset"])] = entry

    models = []
    for key, entry in newest.items():
        metrics = entry.get("metrics") or {}
        models.append(
            {
                "id": key,
                "model_key": entry["model_key"],
                "model_label": entry["model_label"],
                "dataset": entry["dataset"],
                "version": entry["version"],
                "task": entry["task"],
                "threshold": entry.get("threshold"),
                "created_at": entry.get("created_at"),
                "metrics": metrics,
                "explanation_audit": entry.get("explanation_audit") or {},
                "label": f"{entry['model_label']} - {_dataset_label(entry['dataset'])}",
            }
        )
    return sorted(models, key=lambda item: (item["dataset"], item["model_key"]))


def _dataset_label(dataset: str) -> str:
    try:
        from ml.config import DATASETS

        config = DATASETS.get(dataset)
        if config and config.domain:
            return config.domain
    except Exception:  # noqa: BLE001 - a label is never worth an exception
        pass
    return dataset.replace("_", " ").title()


def load_model(model_key: str, dataset: str | None = None) -> LoadedModel:
    """Load a model, reusing the cached copy unless the artefact has changed."""
    entry = registry.latest_entry(model_key, dataset=dataset)
    if entry is None:
        raise ModelUnavailable(
            f"no registered model for '{model_key}'"
            + (f" on dataset '{dataset}'" if dataset else "")
            + ". Train one with `python -m ml.train_all`."
        )

    path = Path(entry["artifact"])
    if not path.exists():
        raise ModelUnavailable(f"the registry points at a missing artefact: {path}")

    identifier = model_id(entry["model_key"], entry["dataset"])
    mtime = path.stat().st_mtime
    cached = _CACHE.get(identifier)
    if cached and cached.artifact_path == path and cached.loaded_mtime == mtime:
        return cached

    with _LOCK:
        cached = _CACHE.get(identifier)
        if cached and cached.artifact_path == path and cached.loaded_mtime == mtime:
            return cached
        LOGGER.info("Loading model artefact %s", path.name)
        bundle = registry.load(entry["model_key"], dataset=entry["dataset"], version=entry["version"])
        loaded = LoadedModel(
            key=identifier, bundle=bundle, artifact_path=path, loaded_mtime=mtime
        )
        _CACHE[identifier] = loaded
        return loaded


def load_by_id(identifier: str) -> LoadedModel:
    """Load from the ``model_key|dataset`` identifier the dropdown submits."""
    if "|" in identifier:
        model_key, dataset = identifier.split("|", 1)
        return load_model(model_key, dataset)
    return load_model(identifier)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def predict(
    frame: pd.DataFrame,
    identifier: str,
    top_drivers: int = 3,
    reference_column: str | None = None,
) -> tuple[list[dict], dict]:
    """Score a frame of customers and recommend an action for each.

    Args:
        frame: Raw customer rows, as uploaded. Columns the model does not know
            are ignored; columns it expects but does not receive are imputed by
            the pipeline, and reported in the metadata so the caller can see how
            much of the input was actually missing.
        identifier: ``model_key|dataset``, from the model selector.
        top_drivers: Drivers surfaced per customer.
        reference_column: Column to echo back as the customer reference.

    Returns:
        ``(results, meta)``. Each result carries the calibrated probability, the
        banded score, the strongest drivers, and the recommended intervention
        with its owner and priority. ``meta`` carries the model version, the
        schema reconciliation and the taxonomy version, all of which belong in
        the audit trail.
    """
    if frame.empty:
        raise ValueError("the uploaded file contains no data rows")

    loaded = load_by_id(identifier)
    bundle = loaded.bundle
    pipeline = bundle["pipeline"]
    classes = np.asarray(bundle["classes"])
    task = bundle["task"]
    threshold = bundle.get("threshold")
    focus_index = int(bundle.get("focus_index", len(classes) - 1))
    focus_label = bundle.get("focus_label", classes[focus_index])
    feature_columns = list(bundle["feature_columns"])
    names = list(bundle["transformed_feature_names"])
    contested = list(bundle.get("contested_features") or [])

    expected = set(feature_columns)
    supplied = set(frame.columns)
    # Derived predictors are computed downstream from the raw columns, so a
    # column that is "missing" only because it is derived is not really missing.
    missing = sorted(
        column for column in expected - supplied
        if column not in _derivable(supplied)
    )
    ignored = sorted(supplied - expected - {reference_column} if reference_column else supplied - expected)

    aligned = align_columns(frame, feature_columns)
    raw_proba = pipeline.predict_proba(aligned)
    proba = apply_calibration(raw_proba, bundle.get("calibrators"))
    predicted = labels_from_proba(
        proba, classes, task, threshold,
        bundle.get("positive_label") if task == "binary" else None,
    )

    matrix = transform_features(pipeline, aligned)
    explainer = loaded.explainer()
    attributions = explanation_layer.shap_matrix(
        explainer, matrix, class_index=focus_index, n_classes=len(classes)
    )

    table = decision.load_rules()
    reference = bundle.get("value_reference")
    values = _customer_values(frame, reference)

    results: list[dict] = []
    for position in range(len(aligned)):
        predicted_label = predicted[position]
        # The risk figure is always the probability of the churn class: that is
        # what the retention team is triaging on. Confidence is the probability
        # of whatever class the model actually assigned, which is a different
        # question and is reported separately rather than conflated.
        probability = float(proba[position, focus_index])
        confidence = float(np.max(proba[position]))

        drivers = explanation_layer.local_drivers(
            attributions[position], matrix[position], names, top_k=top_drivers
        )
        # Mark, at the point of display, every driver the explanation audit found
        # the two explainers disagreeing about. The customer still sees the driver
        # -- hiding it would misrepresent the model -- but it arrives labelled as
        # disputed rather than as settled fact.
        contested_set = set(contested)
        for driver in drivers:
            driver["contested"] = driver["feature"] in contested_set

        recommendation = decision.recommend(
            predicted_label=predicted_label,
            probability=probability,
            drivers=drivers,
            value=values[position],
            contested_features=contested,
            table=table,
        )

        reference_value = None
        if reference_column and reference_column in frame.columns:
            reference_value = str(frame.iloc[position][reference_column])

        results.append(
            {
                "row": position + 1,
                "customer_ref": reference_value or f"row {position + 1}",
                "predicted_label": _native(predicted_label),
                "predicted_churn": bool(str(predicted_label) == str(focus_label)),
                "probability": probability,
                "probability_pct": round(100 * probability, 1),
                "confidence": confidence,
                "confidence_pct": round(100 * confidence, 1),
                "band": recommendation["band"],
                "band_label": recommendation["band_label"],
                "band_colour": recommendation["band_colour"],
                "priority": recommendation["priority"],
                "priority_label": recommendation["priority_label"],
                "priority_score": recommendation["priority_score"],
                "drivers": drivers,
                "recommendation": recommendation,
                "customer_value": values[position],
                "input_data": _row_payload(frame.iloc[position]),
            }
        )

    meta = {
        "model_key": bundle["model_key"],
        "model_label": bundle["model_label"],
        "model_id": loaded.key,
        "model_version": bundle["version"],
        "dataset": bundle["dataset"],
        "task": task,
        "threshold": threshold,
        "focus_label": _native(focus_label),
        "taxonomy_version": table.get("version"),
        "rows": len(results),
        "missing_columns": missing,
        "ignored_columns": ignored,
        "contested_features": contested,
        "metrics": bundle.get("metrics", {}),
        "trained_at": bundle.get("created_at"),
    }
    if missing:
        LOGGER.warning(
            "Scored %d rows with %d expected column(s) absent: %s",
            len(results), len(missing), ", ".join(missing),
        )
    return results, meta


def _derivable(supplied: set[str]) -> set[str]:
    """Columns the feature layer will manufacture from what was supplied.

    Reporting a derived column as "missing" would send a user hunting for a field
    that never existed in their CRM in the first place.
    """
    from ml.features import engineer_features

    probe = pd.DataFrame({column: [0.0] for column in supplied})
    try:
        return set(engineer_features(probe).columns) - supplied
    except Exception:  # noqa: BLE001 - a diagnostic must not break a prediction
        return set()


def _customer_values(frame: pd.DataFrame, reference: dict | None) -> list[float | None]:
    """The value at stake per customer, expressed against the training median.

    Priority scales with this, so that a near-certain churner worth very little
    does not outrank a slightly less certain one worth a great deal.
    """
    if not reference or reference.get("column") not in frame.columns:
        return [None] * len(frame)

    median = float(reference.get("median") or 0.0) or 1.0
    raw = pd.to_numeric(frame[reference["column"]], errors="coerce")
    # Expressed as a multiple of the training median, so one rule table can serve
    # a dataset denominated in account balances and one denominated in monthly fees.
    scaled = (raw / median).replace([np.inf, -np.inf], np.nan)
    return [None if pd.isna(value) else float(value) for value in scaled]


def _row_payload(row: pd.Series) -> dict:
    """The customer's own values, as JSON the audit log can store."""
    payload = {}
    for column, value in row.items():
        payload[str(column)] = _native(value)
    return payload


def _native(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_native(item) for item in value]
    if pd.isna(value):
        return None
    return value if isinstance(value, (int, float, str)) else str(value)


def clear_cache() -> None:
    """Drop every cached model. Used by the tests and after a retrain."""
    with _LOCK:
        _CACHE.clear()
