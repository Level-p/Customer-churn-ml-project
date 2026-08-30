"""Probability calibration.

The decision layer consumes probabilities rather than labels, so a miscalibrated
0.8 is operationally misleading. Ensembles in particular tend to produce
distorted probabilities, so every model is calibrated on out-of-fold predictions
taken from the training partition, never from the test partition.

Isotonic regression is used because it is non-parametric and, being monotone, it
leaves the ranking of records and therefore the ordering of Shapley attributions
untouched: the explanation still describes the model that produced the score.

Only stock scikit-learn estimators are stored in the artefact (a list of fitted
``IsotonicRegression`` objects), so a saved model can be unpickled by the Django
service without importing any bespoke class.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.config import CALIBRATION_FOLDS, RANDOM_SEED


def cross_val_probabilities(
    estimator,
    X,
    y,
    folds: int = CALIBRATION_FOLDS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Out-of-fold probabilities for the training partition.

    The estimator is cloned and refitted inside each fold, so no record is ever
    scored by a model that saw it -- and because the sampler lives inside the
    pipeline, SMOTE runs within the fold rather than across the split. These
    probabilities are the honest validation signal used both to fit the
    calibrators and to choose the operating threshold; the test partition is not
    touched by either decision.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(
        estimator, X, y, cv=splitter, method="predict_proba", n_jobs=None
    )
    return np.asarray(oof_proba, dtype=float)


def fit_calibrators_from_oof(
    oof_proba: np.ndarray,
    y,
    classes: np.ndarray,
) -> list[IsotonicRegression]:
    """Fit one isotonic calibrator per class against out-of-fold probabilities."""
    oof_proba = np.asarray(oof_proba, dtype=float)
    calibrators: list[IsotonicRegression] = []
    for index, label in enumerate(classes):
        target = (np.asarray(y) == label).astype(float)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(oof_proba[:, index], target)
        calibrators.append(calibrator)
    return calibrators


def fit_calibrators(
    estimator,
    X,
    y,
    classes: np.ndarray,
    folds: int = CALIBRATION_FOLDS,
    seed: int = RANDOM_SEED,
) -> list[IsotonicRegression]:
    """Convenience wrapper: generate out-of-fold probabilities and calibrate."""
    oof_proba = cross_val_probabilities(estimator, X, y, folds=folds, seed=seed)
    return fit_calibrators_from_oof(oof_proba, y, classes)


def apply_calibration(proba: np.ndarray, calibrators: list[IsotonicRegression] | None) -> np.ndarray:
    """Map raw probabilities through the fitted calibrators and renormalise.

    Renormalisation is required because per-class isotonic maps do not preserve
    the simplex. If every calibrated score for a row collapses to zero the raw
    row is returned unchanged, which is the conservative fallback.
    """
    proba = np.asarray(proba, dtype=float)
    if not calibrators:
        return proba
    if proba.ndim == 1:  # a single row of class probabilities
        proba = proba.reshape(1, -1)

    calibrated = np.column_stack(
        [calibrator.predict(proba[:, index]) for index, calibrator in enumerate(calibrators)]
    )
    calibrated = np.clip(calibrated, 1e-9, 1.0)
    totals = calibrated.sum(axis=1, keepdims=True)
    degenerate = (totals <= 1e-8).ravel()
    calibrated = calibrated / totals
    if degenerate.any():
        calibrated[degenerate] = proba[degenerate]
    return calibrated
