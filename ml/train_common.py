"""The training protocol shared by every model in the ladder.

One protocol runs for logistic regression, random forest and XGBoost alike, so
that the comparison between them measures the models and not the diligence with
which each was tuned:

 1. prepare the data (clean, encode, scale, drop redundant predictors, split);
 2. tune hyperparameters by stratified cross-validation on the training
    partition, with SMOTE applied inside each fold;
 3. produce out-of-fold probabilities and fit isotonic calibrators on them;
 4. choose the operating threshold on those same validation probabilities, by
    maximising F1 subject to the precision floor the budget holder set;
 5. refit on the full training partition and evaluate once, and only once, on
    the untouched test partition;
 6. explain every scored record with SHAP and audit those explanations against
    LIME on a fixed, risk-stratified sample;
 7. register the artefact with its data snapshot, hyperparameters, evaluation
    report and explanation audit.

Steps 3 to 6 are what the model layer of the framework adds to an ordinary
benchmark script: the model is selected on the joint profile of discrimination,
calibration and explanation quality, not on discrimination alone.
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from ml import decision
from ml import explain as explanation_layer
from ml import fairness
from ml import registry
from ml.calibration import apply_calibration, cross_val_probabilities, fit_calibrators_from_oof
from ml.config import (
    CALIBRATION_FOLDS,
    CV_FOLDS,
    ECONOMICS,
    EXPLANATION_AUDIT_SAMPLE,
    MIN_PRECISION,
    RANDOM_SEED,
    TEST_SIZE,
    TOP_DRIVERS,
    DatasetConfig,
    DATASETS,
    get_dataset,
    get_logger,
    seed_everything,
)
from ml.data_prep import (
    CLASSIFIER_STEP,
    PreparedData,
    build_pipeline,
    prepare,
    transform_features,
    transformed_feature_names,
)
from ml.evaluation import evaluate, labels_from_proba, log_report, select_threshold
from ml.registry import MODEL_LABELS

LOGGER = get_logger("ml.train")


class Tuner(Protocol):
    """Search the hyperparameter space and return the winning configuration."""

    def __call__(
        self,
        pipeline,
        data: PreparedData,
        y_train: np.ndarray,
        args: argparse.Namespace,
        context: dict,
    ) -> tuple[dict, dict]:
        ...


def scoring_for(task: str, focus_index: int = 1):
    """The criterion the hyperparameter search maximises.

    Average precision is used for binary tasks: it is threshold-free and, unlike
    ROC-AUC, it does not flatter a model whose positive class is rare. The
    operating threshold is a separate and later business decision, so the search
    must not be allowed to bake one in.

    ``pos_label`` is pinned to the encoded index of the positive class rather
    than left at scikit-learn's default of 1. Where the positive label happens to
    sort first -- "Churn" before "Retained", say -- the default would quietly
    optimise the wrong class.
    """
    if task != "binary":
        return "f1_macro"
    return make_scorer(
        average_precision_score,
        response_method="predict_proba",
        pos_label=focus_index,
    )


def cross_validator(args: argparse.Namespace) -> StratifiedKFold:
    """The fold structure used by every search in the model layer."""
    return StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def common_parser(description: str) -> argparse.ArgumentParser:
    """Arguments shared by all three training scripts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="bank_churn", choices=sorted(DATASETS),
        help="registered dataset to train on",
    )
    parser.add_argument("--csv", type=Path, help="train on an arbitrary CSV instead")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="stratified subsample of the source data, for quick iteration",
    )
    parser.add_argument("--target", help="label column (required with --csv)")
    parser.add_argument(
        "--task", choices=["binary", "multiclass"],
        help="classification task (inferred from the label when omitted)",
    )
    parser.add_argument(
        "--positive-label", default=None,
        help="label treated as the positive class in a binary task",
    )
    parser.add_argument("--drop", nargs="*", default=None, help="columns to drop (leakage)")
    parser.add_argument("--id-columns", nargs="*", default=None, help="identifier columns")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--explain-rows", type=int, default=2000,
        help="test rows to attribute with SHAP (the audit samples from these)",
    )
    parser.add_argument(
        "--audit-sample", type=int, default=EXPLANATION_AUDIT_SAMPLE,
        help="records on which SHAP is cross-checked against LIME",
    )
    parser.add_argument(
        "--no-audit", action="store_true",
        help="skip the LIME audit (faster; the artefact then carries no audit)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="smoke-test mode: small search space, few folds, short audit",
    )
    return parser


def dataset_from_args(args: argparse.Namespace) -> DatasetConfig:
    """Resolve a registered dataset, or build a configuration for a supplied CSV."""
    if not args.csv:
        config = get_dataset(args.dataset)
        if args.task:
            config.task = args.task
        return config

    if not args.target:
        raise SystemExit("--target is required when --csv is given")

    frame = pd.read_csv(args.csv, nrows=5000)
    if args.target not in frame.columns:
        raise SystemExit(f"target '{args.target}' is not a column of {args.csv}")

    labels = frame[args.target].dropna().unique()
    task = args.task or ("binary" if len(labels) == 2 else "multiclass")
    positive = args.positive_label
    if task == "binary" and positive is None:
        # Default to the rarer label: in this problem family the minority class
        # is the one that costs money to miss.
        counts = frame[args.target].value_counts()
        positive = counts.index[-1]
    elif positive is not None and pd.api.types.is_numeric_dtype(frame[args.target]):
        positive = type(labels[0])(positive)

    return DatasetConfig(
        name=Path(args.csv).stem,
        csv_path=args.csv,
        target=args.target,
        task=task,
        positive_label=positive,
        id_columns=tuple(args.id_columns or ()),
        drop_columns=tuple(args.drop or ()),
        description=f"Ad-hoc dataset loaded from {args.csv}",
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainingOutcome:
    bundle: dict
    entry: dict
    report: object


def train(
    model_key: str,
    build_estimator: Callable[[PreparedData, argparse.Namespace], object],
    tune: Tuner,
    args: argparse.Namespace,
    finalise: Callable[[object, PreparedData, np.ndarray, argparse.Namespace], object] | None = None,
) -> TrainingOutcome:
    """Run the full protocol for one model and register the artefact.

    Args:
        model_key: Registry key, e.g. ``"xgboost"``.
        build_estimator: Returns the unfitted classifier for this model family.
        tune: Searches the hyperparameter space; returns ``(best_params, summary)``
            where the parameter keys are pipeline-qualified, e.g.
            ``"classifier__max_depth"``.
        args: Parsed command-line arguments.
        finalise: Optional hook applied to the tuned pipeline before calibration,
            used by XGBoost to fix the number of boosting rounds by early
            stopping on a held-out validation fold.
    """
    started = time.perf_counter()
    label = MODEL_LABELS.get(model_key, model_key)
    seed_everything(args.seed)

    if args.fast:
        args.cv_folds = min(args.cv_folds, 3)
        args.audit_sample = min(args.audit_sample, 20)
        args.explain_rows = min(args.explain_rows, 400)

    config = dataset_from_args(args)
    _banner(f"1/7  DATA LAYER  |  {label}  |  dataset: {config.name}")
    data = prepare(
        config,
        test_size=args.test_size,
        seed=args.seed,
        max_rows=getattr(args, "max_rows", None),
    )
    value_reference = _value_reference(data)

    # The classifier is fitted on integer-encoded labels: XGBoost requires them,
    # and using them everywhere keeps one probability-column convention across
    # the three models. Column i of every probability matrix in this pipeline is
    # the class ``encoder.classes_[i]``, and nothing downstream may assume more.
    encoder = LabelEncoder().fit(pd.concat([data.y_train, data.y_test]))
    classes = encoder.classes_
    y_train = encoder.transform(data.y_train)
    y_test_labels = data.y_test.to_numpy()

    positive_label = config.positive_label if config.task == "binary" else None
    if config.task == "binary":
        if positive_label not in set(classes.tolist()):
            raise SystemExit(
                f"positive label '{positive_label}' is not one of the observed "
                f"classes {classes.tolist()}"
            )
        focus_label = positive_label
    else:
        preferred = [c for c in (config.class_order or []) if c in set(classes.tolist())]
        focus_label = preferred[-1] if preferred else classes[-1]
    focus_index = int(np.flatnonzero(classes == focus_label)[0])
    LOGGER.info("Classes (probability column order): %s", classes.tolist())
    LOGGER.info("Explanations are reported for class '%s'", focus_label)

    _banner(f"2/7  MODEL LAYER  |  hyperparameter search ({args.cv_folds}-fold, SMOTE in-fold)")
    estimator = build_estimator(data, args)
    pipeline = build_pipeline(
        estimator,
        numeric=data.numeric_columns,
        categorical=data.categorical_columns,
        y_train=y_train,
        seed=args.seed,
    )
    context = {
        "task": config.task,
        "classes": classes,
        "focus_index": focus_index,
        "focus_label": focus_label,
        "scoring": scoring_for(config.task, focus_index),
        "scoring_name": "average_precision" if config.task == "binary" else "f1_macro",
    }
    best_params, cv_summary = tune(pipeline, data, y_train, args, context)
    pipeline.set_params(**best_params)
    LOGGER.info("Best parameters: %s", _strip_prefix(best_params))
    LOGGER.info(
        "Cross-validated %s: %.4f (+/- %.4f)",
        cv_summary.get("scoring", "score"),
        cv_summary.get("best_score", float("nan")),
        cv_summary.get("best_score_std", 0.0),
    )

    if finalise is not None:
        pipeline = finalise(pipeline, data, y_train, args)

    _banner("3/7  MODEL LAYER  |  calibration on out-of-fold probabilities")
    folds = min(CALIBRATION_FOLDS, args.cv_folds)
    oof_raw = cross_val_probabilities(
        clone(pipeline), data.X_train, y_train, folds=folds, seed=args.seed
    )
    calibrators = fit_calibrators_from_oof(oof_raw, y_train, classes)
    oof_calibrated = apply_calibration(oof_raw, calibrators)
    calibration_quality = _calibration_quality(y_train, oof_raw, oof_calibrated, classes, focus_index)
    LOGGER.info(
        "Brier on validation folds: %.4f raw -> %.4f calibrated",
        calibration_quality["brier_raw"], calibration_quality["brier_calibrated"],
    )

    _banner("4/7  DECISION LAYER  |  operating threshold from validation folds")
    threshold_choice = None
    threshold = None
    if config.task == "binary":
        threshold_choice = select_threshold(
            (np.asarray(data.y_train) == positive_label).astype(int),
            oof_calibrated[:, focus_index],
            min_precision=args.min_precision,
        )
        threshold = threshold_choice.threshold
        LOGGER.info(
            "Threshold %.4f (%s): validation precision %.3f, recall %.3f, F1 %.3f",
            threshold_choice.threshold, threshold_choice.rationale,
            threshold_choice.precision, threshold_choice.recall, threshold_choice.f1,
        )
    else:
        LOGGER.info("Multi-class task: the predicted band is the arg-max class.")

    _banner("5/7  MODEL LAYER  |  refit on the full training partition")
    pipeline.fit(data.X_train, y_train)
    fitted = pipeline.named_steps[CLASSIFIER_STEP]
    if not np.array_equal(np.asarray(fitted.classes_), np.arange(len(classes))):
        raise RuntimeError(
            "the fitted classifier's class order does not match the label encoding; "
            "probability columns cannot be trusted"
        )

    _banner("6/7  EVALUATION  |  single pass over the untouched test partition")
    raw_test = pipeline.predict_proba(data.X_test)
    calibrated_test = apply_calibration(raw_test, calibrators)
    report = evaluate(
        y_test_labels,
        calibrated_test,
        classes=classes,
        task=config.task,
        threshold=threshold,
        positive_label=positive_label if config.task == "binary" else None,
        economics=ECONOMICS,
    )
    log_report(label, report)

    _banner("7/7  EXPLANATION LAYER  |  SHAP attributions, audited against LIME")
    explanation = _explain(
        pipeline=pipeline,
        data=data,
        model_key=model_key,
        classes=classes,
        focus_index=focus_index,
        focus_label=focus_label,
        calibrated_test=calibrated_test,
        args=args,
    )

    # The fairness audit runs on the predictions the decision layer would act on,
    # at the threshold it would act at -- not on some other configuration that
    # happens to look better.
    y_pred = labels_from_proba(
        calibrated_test, classes, config.task, threshold,
        positive_label if config.task == "binary" else None,
    )
    fairness_audit = fairness.audit(
        frame=data.X_test,
        y_true=y_test_labels,
        y_pred=y_pred,
        sensitive_attributes=config.sensitive_attributes,
        positive_label=positive_label if config.task == "binary" else focus_label,
        global_importance=explanation["global_importance"],
    )

    elapsed = time.perf_counter() - started
    bundle = {
        "model_key": model_key,
        "model_label": label,
        "dataset": config.name,
        "dataset_domain": config.domain,
        "dataset_description": config.description,
        "version": registry.new_version(),
        "task": config.task,
        "classes": [_native(value) for value in classes],
        "class_order": [_native(value) for value in (config.class_order or classes)],
        "positive_label": _native(positive_label) if positive_label is not None else None,
        "focus_label": _native(focus_label),
        "focus_index": focus_index,
        "target": config.target,
        "threshold": threshold,
        "threshold_choice": threshold_choice.as_dict() if threshold_choice else None,
        "pipeline": pipeline,
        "calibrators": calibrators,
        "background": explanation["background"],
        "feature_columns": data.feature_columns,
        "numeric_columns": data.numeric_columns,
        "categorical_columns": data.categorical_columns,
        "transformed_feature_names": explanation["feature_names"],
        "dropped_columns": data.dropped_columns,
        "hyperparameters": _strip_prefix(best_params),
        "cross_validation": cv_summary,
        "calibration": calibration_quality,
        "metrics": report.headline(),
        "evaluation": report.as_dict(),
        "global_importance": explanation["global_importance"],
        "explanation_audit": explanation["audit"],
        "contested_features": explanation["contested_features"],
        "fairness": fairness_audit,
        "sensitive_attributes": list(config.sensitive_attributes),
        "value_reference": value_reference,
        "training_snapshot": data.snapshot,
        "economics": ECONOMICS.as_dict(),
        "seed": args.seed,
        "training_seconds": round(elapsed, 2),
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    entry = registry.save(bundle)
    LOGGER.info("Trained %s in %.1fs -> version %s", label, elapsed, bundle["version"])
    return TrainingOutcome(bundle=bundle, entry=entry, report=report)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def _explain(
    pipeline,
    data: PreparedData,
    model_key: str,
    classes: np.ndarray,
    focus_index: int,
    focus_label,
    calibrated_test: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Attribute the test predictions and audit the attributions."""
    matrix_train = transform_features(pipeline, data.X_train)
    matrix_test = transform_features(pipeline, data.X_test)
    feature_names = transformed_feature_names(pipeline)
    estimator = pipeline.named_steps[CLASSIFIER_STEP]

    background = explanation_layer.background_sample(matrix_train, seed=args.seed)
    explainer = explanation_layer.build_explainer(estimator, background, model_key)

    limit = min(len(matrix_test), max(args.explain_rows, args.audit_sample))
    subset = matrix_test[:limit]
    attributions = explanation_layer.shap_matrix(
        explainer, subset, class_index=focus_index, n_classes=len(classes)
    )
    global_importance = explanation_layer.global_importance(attributions, feature_names, top_k=20)

    LOGGER.info("Leading drivers of '%s' by mean |SHAP| (top 10):", focus_label)
    for driver in global_importance[:10]:
        LOGGER.info(
            "  %2d. %-38s %-22s %.4f  (%s risk)",
            driver["rank"], driver["label"][:38], f"[{driver['category']}]",
            driver["mean_abs_shap"], driver["direction"],
        )

    audit: dict = {}
    contested: list[str] = []
    if not args.no_audit:
        indices = explanation_layer.stratified_audit_sample(
            calibrated_test[:limit, focus_index], size=args.audit_sample, seed=args.seed
        )
        audit = explanation_layer.lime_audit(
            estimator=estimator,
            background=background,
            audit_matrix=subset[indices],
            shap_attributions=attributions[indices],
            feature_names=feature_names,
            class_names=[str(value) for value in classes],
            class_index=focus_index,
            seed=args.seed,
            num_samples=800 if args.fast else 1500,
        )
        contested = [
            entry["feature"]
            for entry in audit.get("contested_features", [])
            if entry["disagreement_rate"] >= 0.5
        ]
        if contested:
            LOGGER.warning(
                "Features to caveat in the interface (SHAP and LIME disagree): %s",
                ", ".join(contested),
            )
    else:
        LOGGER.warning("LIME audit skipped; this artefact carries no explanation-quality evidence")

    return {
        "background": background,
        "feature_names": feature_names,
        "global_importance": global_importance,
        "audit": audit,
        "contested_features": contested,
    }


def _value_reference(data: PreparedData) -> dict | None:
    """Pick the column that stands in for customer value, and its median.

    The decision layer scales priority by the value at stake, but "value" is
    denominated differently in every dataset -- an account balance in the tens of
    thousands, a monthly fee in the tens. Recording the median of the chosen
    column in the artefact lets the priority rule be written once, in relative
    terms, and still mean the same thing in both domains.
    """
    try:
        table = decision.load_rules()
    except decision.RuleTableError as exc:
        LOGGER.warning("Intervention taxonomy unavailable (%s); priority will not scale by value", exc)
        return None

    candidates = (table.get("value_scaling") or {}).get("value_columns", [])
    for column in candidates:
        if column in data.X_train.columns and pd.api.types.is_numeric_dtype(data.X_train[column]):
            median = float(np.nanmedian(data.X_train[column].to_numpy(dtype=float)))
            if median > 0:
                LOGGER.info("Priority scales with '%s' (median %.2f in training data)", column, median)
                return {"column": column, "median": median}
    LOGGER.info("No customer-value column found; priority will rest on probability alone")
    return None


def _calibration_quality(
    y_train: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    classes: np.ndarray,
    focus_index: int,
) -> dict:
    """Brier score before and after calibration, on the validation folds."""

    def brier(proba: np.ndarray) -> float:
        if len(classes) == 2:
            return float(brier_score_loss((y_train == focus_index).astype(int), proba[:, focus_index]))
        onehot = np.zeros_like(proba)
        onehot[np.arange(len(y_train)), y_train] = 1.0
        return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    before, after = brier(raw), brier(calibrated)
    return {
        "method": "isotonic",
        "folds": CALIBRATION_FOLDS,
        "brier_raw": before,
        "brier_calibrated": after,
        "improvement": before - after,
    }


def _strip_prefix(params: dict) -> dict:
    return {key.replace("classifier__", ""): _native(value) for key, value in params.items()}


def _native(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _banner(text: str) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info(text)
    LOGGER.info("=" * 78)
