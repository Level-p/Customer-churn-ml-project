"""Evaluation: discrimination, calibration and the profit reading.

Accuracy alone is uninformative under imbalance -- a classifier that never
predicts the positive class is 83% accurate on a dataset with a 17% positive
rate -- so the protocol reports a suite of complementary measures:

* precision, recall and F1, which govern wasted budget and missed revenue;
* ROC-AUC for threshold-free discrimination;
* the Matthews correlation coefficient, a more conservative summary of the
  confusion matrix under imbalance;
* the Brier score, because the decision layer consumes probabilities;
* a profit-oriented reading at the operating threshold the decision layer uses.

The operating threshold is chosen on validation data to maximise F1 subject to a
precision floor agreed with the budget holder, not left at the default 0.5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import ECONOMICS, MIN_PRECISION, Economics, get_logger

LOGGER = get_logger(__name__)


@dataclass
class ThresholdChoice:
    """The operating point, and the reason it was chosen."""

    threshold: float
    precision: float
    recall: float
    f1: float
    min_precision: float
    satisfied_floor: bool
    rationale: str

    def as_dict(self) -> dict:
        return asdict(self)


def select_threshold(
    y_true: np.ndarray,
    proba_positive: np.ndarray,
    min_precision: float = MIN_PRECISION,
) -> ThresholdChoice:
    """Maximise F1 subject to ``precision >= min_precision``.

    If no threshold clears the floor, the unconstrained best-F1 threshold is
    returned and flagged, so that the shortfall is visible in the model report
    rather than silently absorbed.
    """
    y_true = np.asarray(y_true).astype(int)
    proba_positive = np.asarray(proba_positive, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true, proba_positive)
    # precision_recall_curve returns one more point than it does thresholds.
    precisions, recalls = precisions[:-1], recalls[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )

    eligible = np.flatnonzero(precisions >= min_precision)
    if eligible.size:
        best = eligible[int(np.argmax(f1_scores[eligible]))]
        satisfied, rationale = True, (
            f"max F1 among thresholds with precision >= {min_precision:.2f}"
        )
    else:
        best = int(np.argmax(f1_scores))
        satisfied, rationale = False, (
            f"no threshold reaches the {min_precision:.2f} precision floor; "
            "fell back to unconstrained max F1"
        )
        LOGGER.warning("Precision floor of %.2f unreachable on validation data", min_precision)

    return ThresholdChoice(
        threshold=float(thresholds[best]),
        precision=float(precisions[best]),
        recall=float(recalls[best]),
        f1=float(f1_scores[best]),
        min_precision=float(min_precision),
        satisfied_floor=satisfied,
        rationale=rationale,
    )


def labels_from_proba(
    proba: np.ndarray,
    classes: np.ndarray,
    task: str,
    threshold: float | None,
    positive_label=1,
) -> np.ndarray:
    """Turn probabilities into labels using the operating threshold."""
    proba = np.asarray(proba, dtype=float)
    if task == "binary" and threshold is not None:
        positive_index = int(np.flatnonzero(np.asarray(classes) == positive_label)[0])
        negative_index = 1 - positive_index
        hit = proba[:, positive_index] >= threshold
        return np.where(hit, classes[positive_index], classes[negative_index])
    return np.asarray(classes)[np.argmax(proba, axis=1)]


def campaign_profit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label=1,
    economics: Economics = ECONOMICS,
) -> dict[str, float]:
    """Value the confusion matrix at the operating threshold.

    This is a profit-oriented reading in the spirit of the expected maximum
    profit criterion: the classifier is judged by the value of the intervention
    campaign it induces, not by its error rate alone.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positive_true = y_true == positive_label
    positive_pred = y_pred == positive_label

    tp = int(np.sum(positive_true & positive_pred))
    fp = int(np.sum(~positive_true & positive_pred))
    fn = int(np.sum(positive_true & ~positive_pred))
    tn = int(np.sum(~positive_true & ~positive_pred))

    total = (
        tp * economics.value_true_positive
        - fp * economics.cost_false_positive
        - fn * economics.cost_false_negative
        + tn * economics.value_true_negative
    )
    targeted = tp + fp
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_value": float(total),
        "value_per_record": float(total / max(len(y_true), 1)),
        "records_targeted": targeted,
        "value_per_targeted_record": float(total / targeted) if targeted else 0.0,
        "economics": economics.as_dict(),
    }


@dataclass
class EvaluationReport:
    """Every number the model layer publishes about one fitted model."""

    task: str
    classes: list
    threshold: float | None
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    mcc: float
    brier: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]
    support: dict[str, int]
    profit: dict[str, float] | None = None
    report_text: str = ""
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def headline(self) -> dict[str, float | None]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "mcc": self.mcc,
            "brier": self.brier,
        }


def evaluate(
    y_true,
    proba: np.ndarray,
    classes: np.ndarray,
    task: str,
    threshold: float | None = None,
    positive_label=1,
    economics: Economics = ECONOMICS,
) -> EvaluationReport:
    """Score one model against a held-out partition."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    y_pred = labels_from_proba(proba, classes, task, threshold, positive_label)

    average = "binary" if task == "binary" else "macro"
    kwargs = {"average": average, "zero_division": 0}
    if task == "binary":
        kwargs["pos_label"] = positive_label

    # ROC-AUC can flatter a model under heavy imbalance, so MCC is reported
    # beside it as the more conservative summary of the same confusion matrix.
    try:
        if task == "binary":
            positive_index = int(np.flatnonzero(classes == positive_label)[0])
            roc_auc = float(roc_auc_score(
                (y_true == positive_label).astype(int), proba[:, positive_index]
            ))
        else:
            roc_auc = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=classes
            ))
    except ValueError as exc:  # a class missing from the partition
        LOGGER.warning("ROC-AUC undefined: %s", exc)
        roc_auc = None

    # Brier score: mean squared error between the probability vector and the
    # one-hot truth, which reduces to the classical Brier score when binary.
    if task == "binary":
        positive_index = int(np.flatnonzero(classes == positive_label)[0])
        brier = float(brier_score_loss(
            (y_true == positive_label).astype(int), proba[:, positive_index]
        ))
    else:
        onehot = np.zeros_like(proba)
        index = {label: position for position, label in enumerate(classes)}
        for row, label in enumerate(y_true):
            onehot[row, index[label]] = 1.0
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    per_class: dict[str, dict[str, float]] = {}
    for label in classes:
        per_class[str(label)] = {
            "precision": float(precision_score(
                y_true, y_pred, labels=[label], average="micro", zero_division=0
            )),
            "recall": float(recall_score(
                y_true, y_pred, labels=[label], average="micro", zero_division=0
            )),
            "f1": float(f1_score(
                y_true, y_pred, labels=[label], average="micro", zero_division=0
            )),
        }

    report = EvaluationReport(
        task=task,
        classes=[_native(label) for label in classes],
        threshold=float(threshold) if threshold is not None else None,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, **kwargs)),
        recall=float(recall_score(y_true, y_pred, **kwargs)),
        f1=float(f1_score(y_true, y_pred, **kwargs)),
        roc_auc=roc_auc,
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        brier=brier,
        confusion_matrix=matrix.astype(int).tolist(),
        per_class=per_class,
        support={str(label): int(np.sum(y_true == label)) for label in classes},
        report_text=classification_report(
            y_true, y_pred, labels=classes, zero_division=0
        ),
    )
    if task == "binary":
        report.profit = campaign_profit(y_true, y_pred, positive_label, economics)
    return report


def _native(value):
    """Convert numpy scalars to JSON-serialisable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def log_report(model_name: str, report: EvaluationReport, logger=LOGGER) -> None:
    """Print the evaluation to the console in a form a reviewer can read."""
    line = "-" * 68
    logger.info(line)
    logger.info("EVALUATION | %s | held-out test partition", model_name)
    logger.info(line)
    if report.threshold is not None:
        logger.info("Operating threshold      : %.4f", report.threshold)
    logger.info("Accuracy                 : %.4f", report.accuracy)
    logger.info("Precision (%s)%s: %.4f", report.task[:5],
                " " * max(1, 12 - len(report.task[:5])), report.precision)
    logger.info("Recall                   : %.4f", report.recall)
    logger.info("F1-score                 : %.4f", report.f1)
    if report.roc_auc is not None:
        logger.info("ROC-AUC                  : %.4f", report.roc_auc)
    logger.info("Matthews corr. coef.     : %.4f", report.mcc)
    logger.info("Brier score (lower=better): %.4f", report.brier)
    logger.info("Confusion matrix (rows=true, cols=predicted, order=%s):",
                report.classes)
    for row in report.confusion_matrix:
        logger.info("    %s", "  ".join(f"{value:6d}" for value in row))
    if report.profit:
        profit = report.profit
        logger.info(
            "Campaign value           : %.2f total | %.3f per record | %d targeted",
            profit["total_value"], profit["value_per_record"], profit["records_targeted"],
        )
    logger.info("Per-class report:\n%s", report.report_text)
    logger.info(line)
