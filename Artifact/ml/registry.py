"""Versioned model registry.

Every trained model is stored together with the snapshot of the data that
trained it, its hyperparameters, its evaluation report and its explanation
audit, and every score served by the dashboard records the model version that
produced it. The bookkeeping is unglamorous but essential: when a decision is
later questioned -- by a manager, an auditor, or the customer concerned -- the
prediction, the explanation and the recommendation can be reproduced exactly as
they stood at the time it was taken.

The artefact contains stock scikit-learn / imbalanced-learn / XGBoost objects
only, so the Django service can unpickle a model without importing any bespoke
estimator class.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ml.config import MODELS_DIR, REPORTS_DIR, get_logger

LOGGER = get_logger(__name__)

MANIFEST_PATH = MODELS_DIR / "registry.json"

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def new_version() -> str:
    """Version stamps are UTC timestamps: sortable, and meaningful to a human."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def artifact_path(model_key: str, dataset: str, version: str) -> Path:
    return MODELS_DIR / f"{model_key}__{dataset}__{version}.joblib"


def report_path(model_key: str, dataset: str, version: str) -> Path:
    return REPORTS_DIR / f"{model_key}__{dataset}__{version}.json"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _empty_manifest() -> dict[str, Any]:
    return {"schema": 1, "updated_at": None, "entries": []}


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return _empty_manifest()
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.error("Registry manifest unreadable (%s); starting a fresh one", exc)
        return _empty_manifest()
    manifest.setdefault("entries", [])
    return manifest


def _write_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    temporary.replace(MANIFEST_PATH)  # atomic: a reader never sees a half-file


def list_entries(dataset: str | None = None, model_key: str | None = None) -> list[dict]:
    """Registry entries, newest first."""
    entries = load_manifest()["entries"]
    if dataset:
        entries = [entry for entry in entries if entry["dataset"] == dataset]
    if model_key:
        entries = [entry for entry in entries if entry["model_key"] == model_key]
    return sorted(entries, key=lambda entry: entry["version"], reverse=True)


def latest_entry(model_key: str, dataset: str | None = None) -> dict | None:
    entries = list_entries(dataset=dataset, model_key=model_key)
    return entries[0] if entries else None


def latest_per_model(dataset: str | None = None) -> dict[str, dict]:
    """The newest version of each model key: what the dashboard offers."""
    newest: dict[str, dict] = {}
    for entry in sorted(list_entries(dataset=dataset), key=lambda item: item["version"]):
        newest[entry["model_key"]] = entry
    return newest


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


def save(bundle: dict[str, Any]) -> dict[str, Any]:
    """Persist an artefact and register it. Returns the manifest entry."""
    required = {"model_key", "dataset", "version", "pipeline", "task", "classes"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"artefact is missing required keys: {sorted(missing)}")

    model_key, dataset, version = bundle["model_key"], bundle["dataset"], bundle["version"]
    bundle.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    bundle.setdefault("python_version", platform.python_version())

    destination = artifact_path(model_key, dataset, version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)

    metrics = bundle.get("metrics", {}) or {}
    headline = {
        key: metrics.get(key)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "brier")
    }
    entry = {
        "model_key": model_key,
        "model_label": MODEL_LABELS.get(model_key, model_key.replace("_", " ").title()),
        "dataset": dataset,
        "version": version,
        "task": bundle["task"],
        "classes": [str(label) for label in bundle["classes"]],
        "threshold": bundle.get("threshold"),
        "artifact": str(destination),
        "report": str(report_path(model_key, dataset, version)),
        "created_at": bundle["created_at"],
        "metrics": headline,
        "explanation_audit": {
            key: value
            for key, value in (bundle.get("explanation_audit") or {}).items()
            if key in {"mean_top_k_overlap", "mean_order_agreement", "lime_stability"}
        },
        "training_seconds": bundle.get("training_seconds"),
    }

    manifest = load_manifest()
    manifest["entries"] = [
        existing
        for existing in manifest["entries"]
        if not (
            existing["model_key"] == model_key
            and existing["dataset"] == dataset
            and existing["version"] == version
        )
    ]
    manifest["entries"].append(entry)
    _write_manifest(manifest)

    # The human-readable report travels beside the artefact, not inside it.
    report = report_path(model_key, dataset, version)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: value
                for key, value in bundle.items()
                if key not in {"pipeline", "calibrators", "background", "explainer"}
            },
            handle,
            indent=2,
            default=str,
        )

    LOGGER.info("Saved artefact -> %s", destination)
    LOGGER.info("Saved report   -> %s", report)
    return entry


def load(model_key: str, dataset: str | None = None, version: str | None = None) -> dict[str, Any]:
    """Load one artefact; defaults to the newest version of the model key."""
    entries = list_entries(dataset=dataset, model_key=model_key)
    if version:
        entries = [entry for entry in entries if entry["version"] == version]
    if not entries:
        raise FileNotFoundError(
            f"no registered model for key='{model_key}' dataset='{dataset}' "
            f"version='{version}'. Train one with `python -m ml.train_all`."
        )
    entry = entries[0]
    path = Path(entry["artifact"])
    if not path.exists():
        raise FileNotFoundError(f"registry points at a missing artefact: {path}")
    bundle = joblib.load(path)
    bundle["_entry"] = entry
    return bundle
