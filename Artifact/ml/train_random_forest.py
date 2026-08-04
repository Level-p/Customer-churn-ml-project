"""Train the random forest.

A random forest aggregates several hundred decorrelated decision trees, each
grown on a bootstrap sample with a random subset of features considered at every
split. It captures non-linearities and interactions without manual engineering,
tolerates outliers and the mixed feature types typical of commercial data, and
its bagged structure resists overfitting.

The costs are the ones the framework exists to manage: direct interpretability is
gone, importance is diluted across correlated predictors, and the probability
estimates usually need recalibrating before a decision layer can consume them.
The first is answered by SHAP, the second by the correlation filter in the data
layer, and the third by the isotonic calibration in the training protocol.

Usage:
    python -m ml.train_random_forest
    python -m ml.train_random_forest --dataset retail_demand_bands
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ml.config import get_logger
from ml.data_prep import CLASSIFIER_STEP, PreparedData, transformed_feature_names
from ml.features import friendly_name
from ml.train_common import TrainingOutcome, common_parser, cross_validator, train

MODEL_KEY = "random_forest"
LOGGER = get_logger("ml.train.random_forest")


def build_estimator(data: PreparedData, args: argparse.Namespace) -> RandomForestClassifier:
    """Trees are grown single-threaded; the grid search owns the parallelism.

    Letting both the forest and the search claim every core oversubscribes the
    machine and makes the search slower, not faster.
    """
    return RandomForestClassifier(
        n_estimators=300,
        random_state=args.seed,
        n_jobs=1,
        bootstrap=True,
    )


def tune(
    pipeline,
    data: PreparedData,
    y_train: np.ndarray,
    args: argparse.Namespace,
    context: dict,
) -> tuple[dict, dict]:
    """Grid search over the four parameters that govern a forest's capacity.

    The number of estimators, the depth of each tree, the minimum samples
    required to split, and the number of features offered at each split: between
    them these set how much of the training data the forest is allowed to
    memorise.

    ``class_weight`` is deliberately *not* in the grid. SMOTE has already
    rebalanced every training fold, and weighting the minority class on top of
    that corrects the same imbalance twice -- it doubled the search cost here for
    no measurable gain. A dataset where resampling is inappropriate should turn
    SMOTE off and reinstate the weighting, rather than run both.
    """
    grid = (
        {
            "classifier__n_estimators": [150],
            "classifier__max_depth": [12],
            "classifier__min_samples_split": [2],
            "classifier__max_features": ["sqrt"],
        }
        if args.fast
        else {
            "classifier__n_estimators": [300, 600],
            "classifier__max_depth": [None, 16],
            "classifier__min_samples_split": [2, 10],
            "classifier__max_features": ["sqrt", 0.4],
        }
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=context["scoring"],
        cv=cross_validator(args),
        n_jobs=args.n_jobs,
        refit=False,
        verbose=1,
    )
    candidates = int(np.prod([len(values) for values in grid.values()]))
    LOGGER.info("Grid: %d candidates x %d folds", candidates, args.cv_folds)
    search.fit(data.X_train, y_train)

    index = int(search.best_index_)
    summary = {
        "scoring": context["scoring_name"],
        "folds": args.cv_folds,
        "best_score": float(search.cv_results_["mean_test_score"][index]),
        "best_score_std": float(search.cv_results_["std_test_score"][index]),
        "candidates": int(len(search.cv_results_["params"])),
    }
    return search.best_params_, summary


def log_impurity_importance(outcome: TrainingOutcome, top_k: int = 15) -> None:
    """Print impurity importance beside the Shapley ranking, and mind the gap.

    Mean decrease in impurity is biased towards high-cardinality and continuous
    predictors and says nothing about the direction of an effect, which is why
    the framework does not let it reach a business user. It is logged here as a
    diagnostic: a predictor that the forest splits on constantly but that SHAP
    ranks nowhere is usually a sign of a leak or of a correlated pair that the
    data layer should have caught.
    """
    pipeline = outcome.bundle["pipeline"]
    model: RandomForestClassifier = pipeline.named_steps[CLASSIFIER_STEP]
    names = transformed_feature_names(pipeline)
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1][:top_k]

    LOGGER.info("")
    LOGGER.info("Impurity importance (diagnostic only; top %d):", top_k)
    for rank, index in enumerate(order, start=1):
        LOGGER.info(
            "  %2d. %-42s %.4f", rank, friendly_name(names[index], names)[:42], importances[index]
        )

    shap_top = {driver["feature"] for driver in outcome.bundle["global_importance"][:top_k]}
    impurity_top = {names[index] for index in order}
    overlap = len(shap_top & impurity_top) / max(len(shap_top), 1)
    LOGGER.info(
        "Top-%d agreement between impurity importance and mean |SHAP|: %.0f%%",
        top_k, 100 * overlap,
    )
    if overlap < 0.5:
        LOGGER.warning(
            "The two importance measures disagree substantially. Investigate before "
            "presenting either to a business user."
        )


def main() -> TrainingOutcome:
    parser = common_parser("Train the random forest for the BDSS.")
    args = parser.parse_args()

    outcome = train(
        model_key=MODEL_KEY,
        build_estimator=build_estimator,
        tune=tune,
        args=args,
    )
    log_impurity_importance(outcome)
    return outcome


if __name__ == "__main__":
    main()
