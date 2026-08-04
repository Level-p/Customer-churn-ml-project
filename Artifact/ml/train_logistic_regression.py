"""Train the logistic regression baseline.

Logistic regression models the probability of the positive class as

    p(x) = 1 / (1 + exp(-(b0 + b'x)))

Its advantages are exact interpretability -- every coefficient is a log-odds
contribution that can be read directly, without any post-hoc machinery -- along
with fast training, stable behaviour on modest samples and well-calibrated
probabilities. Its limitation is the linearity assumption: interactions and
threshold effects, which are common in demand data, must be hand-crafted or are
missed.

It is retained not despite that simplicity but because of it. It is the reference
against which the cost of opacity is priced: if the ensembles cannot beat it by a
margin that matters to the business, the business should be running this.

Usage:
    python -m ml.train_logistic_regression
    python -m ml.train_logistic_regression --dataset retail_demand_bands
    python -m ml.train_logistic_regression --csv data/my_sales.csv --target churned
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from ml.config import get_logger
from ml.data_prep import CLASSIFIER_STEP, PreparedData, transformed_feature_names
from ml.features import friendly_name
from ml.train_common import TrainingOutcome, common_parser, cross_validator, train

MODEL_KEY = "logistic_regression"
LOGGER = get_logger("ml.train.logistic_regression")


def build_estimator(data: PreparedData, args: argparse.Namespace) -> LogisticRegression:
    """An L2-penalised linear model; lbfgs converges reliably on scaled inputs.

    L2 is scikit-learn's default penalty and is left implicit: passing it
    explicitly is deprecated from 1.8 onward. The strength of that penalty, C, is
    what the grid search below actually tunes.
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        random_state=args.seed,
        n_jobs=None,  # lbfgs is single-threaded; the grid search parallelises instead
    )


def tune(
    pipeline,
    data: PreparedData,
    y_train: np.ndarray,
    args: argparse.Namespace,
    context: dict,
) -> tuple[dict, dict]:
    """Grid search the inverse regularisation strength under an L2 penalty.

    ``class_weight`` is searched alongside it. SMOTE already rebalances the
    training folds, but on some datasets the residual cost asymmetry is better
    handled by weighting, and the search is allowed to decide which.
    """
    grid = (
        {"classifier__C": [0.1, 1.0], "classifier__class_weight": [None]}
        if args.fast
        else {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=context["scoring"],
        cv=cross_validator(args),
        n_jobs=args.n_jobs,
        refit=False,  # the caller refits on the full training partition
        verbose=1,
    )
    LOGGER.info("Grid: %s", {key.replace("classifier__", ""): value for key, value in grid.items()})
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


def log_coefficients(outcome: TrainingOutcome, top_k: int = 15) -> None:
    """Print the log-odds contributions: the model explaining itself.

    This is the property the ensembles have to buy back with SHAP, and printing
    it beside the Shapley ranking is the cheapest sanity check in the pipeline --
    a linear coefficient and a Shapley value that disagree about the sign of a
    driver is a signal worth chasing before anyone trusts the score.
    """
    pipeline = outcome.bundle["pipeline"]
    model: LogisticRegression = pipeline.named_steps[CLASSIFIER_STEP]
    names = transformed_feature_names(pipeline)

    coefficients = np.asarray(model.coef_)
    if coefficients.shape[0] == 1:  # binary
        rows = [(coefficients[0], str(outcome.bundle["focus_label"]))]
    else:
        rows = [
            (coefficients[index], str(label))
            for index, label in enumerate(outcome.bundle["classes"])
        ]

    for vector, label in rows:
        order = np.argsort(np.abs(vector))[::-1][:top_k]
        LOGGER.info("")
        LOGGER.info("Log-odds coefficients for class '%s' (top %d):", label, top_k)
        LOGGER.info("  %-42s %10s %10s", "predictor", "coef", "odds x")
        for index in order:
            coefficient = float(vector[index])
            LOGGER.info(
                "  %-42s %10.4f %10.3f",
                friendly_name(names[index], names)[:42],
                coefficient,
                float(np.exp(coefficient)),
            )
    LOGGER.info("Intercept: %s", np.round(np.asarray(model.intercept_), 4).tolist())


def main() -> TrainingOutcome:
    parser = common_parser("Train the logistic regression baseline for the BDSS.")
    args = parser.parse_args()

    outcome = train(
        model_key=MODEL_KEY,
        build_estimator=build_estimator,
        tune=tune,
        args=args,
    )
    log_coefficients(outcome)
    return outcome


if __name__ == "__main__":
    main()
