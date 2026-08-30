"""Central configuration for the BDSS machine learning pipeline.

Every path, seed and protocol constant used by the training scripts and by the
Django inference layer is declared here, so that a run can be reproduced
exactly from the values recorded in a model artefact.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BDSS_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("BDSS_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("BDSS_REPORTS_DIR", PROJECT_ROOT / "reports"))
DECISION_DIR = Path(os.environ.get("BDSS_DECISION_DIR", PROJECT_ROOT / "decision"))
RULES_PATH = DECISION_DIR / "intervention_rules.json"

for _directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR, DECISION_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility and training protocol
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42

#: Fraction of records held out for the single, final evaluation.
TEST_SIZE = 0.2

#: Folds used for hyperparameter search. SMOTE is applied *inside* each fold.
CV_FOLDS = 10

#: Folds used to produce out-of-fold probabilities for calibration.
CALIBRATION_FOLDS = 5

#: Precision floor for threshold selection. Precision governs how much of the
#: intervention budget is spent on records that never needed the intervention,
#: so the floor is a business input, not a statistical one.
MIN_PRECISION = 0.60

#: Size of the fixed sample on which SHAP explanations are audited against LIME.
EXPLANATION_AUDIT_SAMPLE = 100

#: Rows of transformed training data stored with each artefact and used as the
#: background distribution when explanations are recomputed at inference time.
SHAP_BACKGROUND_SIZE = 200

#: Correlation above which one member of the pair is dropped: strongly
#: correlated predictors destabilise logistic regression coefficients and
#: dilute Shapley attributions across duplicates.
CORRELATION_THRESHOLD = 0.90

#: Predictors whose variance falls below this value carry no signal.
NEAR_ZERO_VARIANCE = 1e-8

#: Number of drivers surfaced per record in the decision layer.
TOP_DRIVERS = 3


@dataclass(frozen=True)
class Economics:
    """Unit economics for the profit-oriented reading of a classifier.

    A churn model is worth what the retention campaign it induces is worth, so
    the confusion matrix is valued rather than merely counted:

    * a true positive is a churner correctly flagged, offered a retention
      incentive, and kept: the customer's residual value, net of the offer;
    * a false positive is an incentive spent on a customer who was never
      leaving: the cost of the offer, wasted;
    * a false negative is a churner who walks out unnoticed: the customer's
      value, lost;
    * a true negative costs nothing and earns nothing.

    These figures are illustrative placeholders. They are the single most
    important thing an adopting firm must replace with its own numbers, because
    they -- not the model -- decide where the operating threshold lands.
    """

    value_true_positive: float = 100.0   # retained customer value, net of offer
    cost_false_positive: float = 15.0    # retention offer wasted on a loyal customer
    cost_false_negative: float = 200.0   # customer lost, and their revenue with them
    value_true_negative: float = 0.0     # business as usual

    def as_dict(self) -> dict[str, float]:
        return {
            "value_true_positive": self.value_true_positive,
            "cost_false_positive": self.cost_false_positive,
            "cost_false_negative": self.cost_false_negative,
            "value_true_negative": self.value_true_negative,
        }


ECONOMICS = Economics()


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


@dataclass
class DatasetConfig:
    """Declarative description of a tabular dataset.

    Attributes:
        name: Registry key for the dataset.
        csv_path: Location of the source CSV.
        target: Name of the label column.
        task: ``"binary"`` or ``"multiclass"``.
        positive_label: Label treated as the positive class in a binary task.
        class_order: Optional display order for the classes.
        id_columns: Identifiers dropped before modelling and used to detect
            duplicate records.
        drop_columns: Columns removed for leakage, fairness or data-protection
            reasons before the model ever sees them.
        sensitive_attributes: Columns across which the fairness audit compares
            error rates. They may still be used as predictors -- the audit exists
            precisely to test whether they should be.
        domain: Sector, used when comparing behaviour across domains.
        description: Human readable summary shown in the dashboard.
        unlabelled_csv: Optional companion file with no label column, used to
            demonstrate batch scoring through the dashboard.
    """

    name: str
    csv_path: Path
    target: str
    task: str = "binary"
    positive_label: object = 1
    class_order: Sequence[object] | None = None
    id_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    sensitive_attributes: tuple[str, ...] = ()
    domain: str = ""
    description: str = ""
    unlabelled_csv: Path | None = None

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        if self.unlabelled_csv:
            self.unlabelled_csv = Path(self.unlabelled_csv)
        if self.task not in {"binary", "multiclass"}:
            raise ValueError(f"unsupported task '{self.task}'")


DATASETS_DIR = Path(os.environ.get("BDSS_DATASETS_DIR", PROJECT_ROOT / "datasets"))

DATASETS: dict[str, DatasetConfig] = {
    # Retail banking. 165,034 clients, 21.2% exited: the class imbalance the
    # framework is built to handle, at a scale that makes the training protocol
    # earn its keep.
    "bank_churn": DatasetConfig(
        name="bank_churn",
        csv_path=DATASETS_DIR / "Bank" / "train.csv",
        target="Exited",
        task="binary",
        positive_label=1,
        class_order=(0, 1),
        # 'Surname' is dropped rather than merely unused. It is a direct
        # identifier and a proxy for ethnicity and nationality, and no model in
        # a retention system has any business reading one.
        id_columns=("id", "CustomerId"),
        drop_columns=("Surname",),
        sensitive_attributes=("Gender", "Geography", "Age"),
        domain="Retail banking",
        description=(
            "Retail banking customers. The positive class is a client who has "
            "closed their account."
        ),
        unlabelled_csv=DATASETS_DIR / "Bank" / "test.csv",
    ),
    # Subscription streaming: the contractual, subscription-services setting the
    # research is addressed to. Small, and close to balanced, which makes it the
    # useful counterweight to the banking data.
    "netflix_churn": DatasetConfig(
        name="netflix_churn",
        csv_path=DATASETS_DIR / "netflix" / "netflix_customer_churn.csv",
        target="churned",
        task="binary",
        positive_label=1,
        class_order=(0, 1),
        id_columns=("customer_id",),
        drop_columns=(),
        sensitive_attributes=("gender", "region", "age"),
        domain="Subscription streaming",
        description=(
            "Streaming subscribers; the positive class has cancelled. TREAT THE "
            "METRICS WITH SUSPICION: the label is synthetic. A depth-5 decision "
            "tree recovers 90% of it from three columns, so the near-perfect "
            "scores here measure how well a model can rediscover the rule that "
            "generated the file, not how well it would predict a real "
            "cancellation. Use this dataset to exercise the framework, and the "
            "banking dataset to judge it."
        ),
    ),
    # The e-commerce file in datasets/Eshop carries no label column, so it
    # supports no supervised model and is deliberately not registered here. See
    # the limitations section of the README: a churn label defined from
    # `days_since_last_purchase` would either leak through `engagement_score`
    # (r = -0.85 with the cutoff column) or, with both removed, leave predictors
    # that are uncorrelated with the label. Fabricating a third domain is worse
    # than reporting two.
}


def get_dataset(name: str) -> DatasetConfig:
    """Return a registered dataset configuration."""
    try:
        return DATASETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset '{name}'. Known datasets: {known}") from exc


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a console logger configured once per process."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Fix every random seed the pipeline depends on."""
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
