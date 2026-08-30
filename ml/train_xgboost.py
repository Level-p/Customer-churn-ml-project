"""Train the gradient-boosted ensemble.

XGBoost implements regularised gradient boosting: trees are built sequentially,
each correcting the residual errors of its predecessors, with explicit penalties
on tree complexity. It is the strongest performer on tabular data of this kind,
and it handles class imbalance through instance weighting as well as resampling.
Its costs are sensitivity to hyperparameters, a greater risk of overfitting a
small dataset than bagging carries, and the same native opacity as any large
ensemble.

Its parameter space is too large for exhaustive search, so it is tuned by
Bayesian optimisation with Optuna over the learning rate, tree depth, the
subsampling ratios, the regularisation terms and the instance weighting, and the
number of boosting rounds is then fixed by early stopping on a validation fold
carved out of the training partition. The test partition is not consulted at any
point in that process.

Deep neural architectures were considered and set aside: on tabular data of this
size they seldom beat a tuned gradient boosting model, while demanding more data,
more tuning and heavier explanation machinery. Every model retained here is
compatible with exact Shapley computation -- logistic regression trivially, the
two ensembles through TreeExplainer -- which is what keeps the explanation layer
honest rather than aspirational.

Usage:
    python -m ml.train_xgboost
    python -m ml.train_xgboost --trials 50
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from ml.config import get_logger
from ml.data_prep import (
    CLASSIFIER_STEP,
    PREPROCESS_STEP,
    SAMPLER_STEP,
    VARIANCE_STEP,
    PreparedData,
    transformed_feature_names,
)
from ml.features import friendly_name
from ml.train_common import TrainingOutcome, common_parser, cross_validator, train

MODEL_KEY = "xgboost"
LOGGER = get_logger("ml.train.xgboost")

#: Cap for the early-stopping search. The winning round count is almost always
#: far below it; the cap exists so a pathological configuration cannot run away.
MAX_BOOSTING_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
VALIDATION_FRACTION = 0.15


def build_estimator(data: PreparedData, args: argparse.Namespace) -> XGBClassifier:
    """The histogram tree method is exact enough here and markedly faster."""
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=2,  # the Optuna trials and the CV folds already saturate the machine
        eval_metric="logloss",
        verbosity=0,
    )


def _search_space(trial: optuna.Trial, task: str, imbalance_ratio: float) -> dict:
    """The parameter space Optuna searches.

    ``scale_pos_weight`` is bounded above by the observed imbalance ratio: SMOTE
    has already rebalanced the training folds, so weighting the positive class by
    the full ratio on top of that would double-count the correction and push the
    model into recall at any price.
    """
    space = {
        "classifier__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classifier__max_depth": trial.suggest_int("max_depth", 3, 10),
        "classifier__min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "classifier__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "classifier__gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "classifier__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "classifier__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "classifier__n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
    }
    if task == "binary":
        space["classifier__scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight", 1.0, max(1.0, min(imbalance_ratio, 4.0))
        )
    return space


def tune(
    pipeline,
    data: PreparedData,
    y_train: np.ndarray,
    args: argparse.Namespace,
    context: dict,
) -> tuple[dict, dict]:
    """Bayesian optimisation over the boosting parameters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials = 5 if args.fast else int(getattr(args, "trials", 30))

    counts = np.bincount(y_train.astype(int))
    imbalance_ratio = float(counts.max() / max(counts.min(), 1))
    splitter = cross_validator(args)

    def objective(trial: optuna.Trial) -> float:
        candidate = clone(pipeline)
        candidate.set_params(**_search_space(trial, context["task"], imbalance_ratio))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                candidate,
                data.X_train,
                y_train,
                scoring=context["scoring"],
                cv=splitter,
                n_jobs=args.n_jobs,
                error_score="raise",
            )
        trial.set_user_attr("std", float(np.std(scores)))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=MODEL_KEY)
    LOGGER.info(
        "Optuna: %d trials, %d-fold CV, objective=%s, imbalance ratio %.2f",
        trials, args.cv_folds, context["scoring_name"], imbalance_ratio,
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    for trial in sorted(study.trials, key=lambda t: t.value or -np.inf, reverse=True)[:5]:
        LOGGER.info("  trial %2d: %.4f  %s", trial.number, trial.value or float("nan"), trial.params)

    best_params = {f"classifier__{key}": value for key, value in study.best_params.items()}
    summary = {
        "scoring": context["scoring_name"],
        "folds": args.cv_folds,
        "best_score": float(study.best_value),
        "best_score_std": float(study.best_trial.user_attrs.get("std", 0.0)),
        "candidates": trials,
        "sampler": "TPE (Bayesian)",
    }
    return best_params, summary


def finalise(pipeline, data: PreparedData, y_train: np.ndarray, args: argparse.Namespace):
    """Fix the number of boosting rounds by early stopping on a validation fold.

    A fold is carved out of the *training* partition; the preprocessing and the
    sampler are fitted on the remainder only, so the validation fold is scored by
    a model that has never seen it, directly or through a synthetic neighbour.
    The winning round count is written back into the pipeline, which the caller
    then refits on the whole training partition.
    """
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        data.X_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        stratify=y_train,
        random_state=args.seed,
    )

    probe = clone(pipeline)
    preprocessor = probe.named_steps[PREPROCESS_STEP]
    variance = probe.named_steps[VARIANCE_STEP]
    matrix_fit = variance.fit_transform(preprocessor.fit_transform(X_fit))
    matrix_validation = variance.transform(preprocessor.transform(X_validation))

    sampler = probe.named_steps.get(SAMPLER_STEP)
    if sampler is not None and sampler != "passthrough":
        matrix_fit, y_fit = sampler.fit_resample(matrix_fit, y_fit)

    model: XGBClassifier = probe.named_steps[CLASSIFIER_STEP]
    model.set_params(
        n_estimators=MAX_BOOSTING_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(matrix_fit, y_fit, eval_set=[(matrix_validation, y_validation)], verbose=False)

    rounds = int(getattr(model, "best_iteration", MAX_BOOSTING_ROUNDS - 1)) + 1
    LOGGER.info(
        "Early stopping on a %.0f%% validation fold: %d boosting rounds (from a cap of %d)",
        100 * VALIDATION_FRACTION, rounds, MAX_BOOSTING_ROUNDS,
    )
    pipeline.set_params(
        classifier__n_estimators=rounds, classifier__early_stopping_rounds=None
    )
    return pipeline


def log_feature_importance(outcome: TrainingOutcome, top_k: int = 15) -> None:
    """Feature importance analysis: native gain, and how it differs from SHAP.

    Gain measures how much a feature improved the loss at the splits that used
    it, aggregated over the ensemble. It is a global, unsigned, model-internal
    quantity: it cannot tell a planner whether a driver pushed *this* SKU's
    forecast up or down, and it is known to spread credit unevenly across
    correlated features. Shapley values answer both questions, which is why they
    and not gain are what the decision layer consumes. Gain is logged here
    because a large divergence between the two rankings is diagnostic.
    """
    pipeline = outcome.bundle["pipeline"]
    model: XGBClassifier = pipeline.named_steps[CLASSIFIER_STEP]
    names = transformed_feature_names(pipeline)
    booster = model.get_booster()

    LOGGER.info("")
    LOGGER.info("Native feature importance (top %d):", top_k)
    LOGGER.info("  %-40s %10s %10s %8s", "predictor", "gain", "cover", "splits")

    scores = {
        metric: booster.get_score(importance_type=metric)
        for metric in ("total_gain", "total_cover", "weight")
    }
    # The booster names features f0, f1, ... in the order it received them.
    ranked = sorted(
        scores["total_gain"].items(), key=lambda item: item[1], reverse=True
    )[:top_k]
    gain_top: list[str] = []
    for key, gain in ranked:
        index = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else -1
        name = names[index] if 0 <= index < len(names) else key
        gain_top.append(name)
        LOGGER.info(
            "  %-40s %10.1f %10.1f %8d",
            friendly_name(name, names)[:40],
            gain,
            scores["total_cover"].get(key, 0.0),
            int(scores["weight"].get(key, 0)),
        )

    shap_top = [driver["feature"] for driver in outcome.bundle["global_importance"][:top_k]]
    overlap = len(set(gain_top) & set(shap_top)) / max(len(shap_top), 1)
    LOGGER.info(
        "Top-%d agreement between gain and mean |SHAP|: %.0f%%", top_k, 100 * overlap
    )
    only_gain = [friendly_name(name, names) for name in gain_top if name not in set(shap_top)]
    if only_gain:
        LOGGER.info(
            "Split on often but attributed little by SHAP: %s", ", ".join(only_gain[:5])
        )
    LOGGER.info(
        "Gain is a global, unsigned diagnostic. Only the Shapley attributions reach "
        "the dashboard, because only they carry a direction and a per-record value."
    )


def main() -> TrainingOutcome:
    parser = common_parser("Train the XGBoost model for the BDSS.")
    parser.add_argument(
        "--trials", type=int, default=30, help="Optuna trials in the Bayesian search"
    )
    args = parser.parse_args()

    outcome = train(
        model_key=MODEL_KEY,
        build_estimator=build_estimator,
        tune=tune,
        args=args,
        finalise=finalise,
    )
    log_feature_importance(outcome)
    return outcome


if __name__ == "__main__":
    main()
