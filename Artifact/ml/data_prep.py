"""Data layer: loading, cleaning, encoding, scaling, splitting and resampling.

The same protocol runs on every dataset so that cross-domain comparisons are not
confounded by differing preprocessing:

* duplicate identifiers are removed;
* continuous fields are median-imputed, categorical fields receive an explicit
  "Missing" level;
* categorical predictors are one-hot encoded, continuous predictors are min-max
  scaled to the unit interval;
* near-zero-variance predictors are dropped, and one member of any pair with
  absolute correlation above 0.9 is removed;
* the data is split 80/20, stratified on the label;
* SMOTE sits *inside* the modelling pipeline, so when the pipeline is
  cross-validated the oversampling happens within each fold and no synthetic
  point derived from a validation record can leak into the fold that scores it.
  The test partition keeps its natural class distribution throughout.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml import features as feature_layer
from ml.config import (
    CORRELATION_THRESHOLD,
    NEAR_ZERO_VARIANCE,
    RANDOM_SEED,
    TEST_SIZE,
    DatasetConfig,
    get_logger,
)

LOGGER = get_logger(__name__)

PREPROCESS_STEP = "preprocess"
VARIANCE_STEP = "variance"
SAMPLER_STEP = "smote"
CLASSIFIER_STEP = "classifier"


@dataclass
class PreparedData:
    """Everything the model layer needs from the data layer."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]
    classes: np.ndarray
    dropped_columns: dict[str, list[str]] = field(default_factory=dict)
    snapshot: dict[str, object] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.numeric_columns) + list(self.categorical_columns)


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #


def dataset_fingerprint(path: Path) -> dict[str, object]:
    """Hash the source CSV so a model can be tied to the data that trained it."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV, treating the usual spellings of "missing" as missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}. Run `python -m ml.generate_dataset` "
            "to build the bundled retail demand dataset."
        )
    frame = pd.read_csv(
        path,
        na_values=["", " ", "NA", "N/A", "na", "null", "NULL", "?", "-"],
        keep_default_na=True,
    )
    LOGGER.info("Loaded %s: %d rows x %d columns", path.name, len(frame), frame.shape[1])
    return frame


def clean(frame: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Drop duplicates, identifiers and leakage columns; coerce numeric fields."""
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]

    if config.target not in df.columns:
        raise KeyError(f"target column '{config.target}' is absent from the dataset")

    identifiers = [column for column in config.id_columns if column in df.columns]
    if identifiers:
        before = len(df)
        df = df.drop_duplicates(subset=identifiers, keep="first")
        if before != len(df):
            LOGGER.info("Removed %d duplicate identifier rows", before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        LOGGER.info("Removed %d fully duplicated rows", before - len(df))

    df = df.dropna(subset=[config.target])

    leakage = [column for column in config.drop_columns if column in df.columns]
    if leakage:
        LOGGER.info("Dropping leakage columns: %s", ", ".join(leakage))
    df = df.drop(columns=identifiers + leakage, errors="ignore")

    # Fields such as Telco's TotalCharges arrive as text with stray blanks. Under
    # pandas 3 a text column is a str dtype rather than object, so the test is on
    # "not numeric" rather than on a specific storage dtype.
    for column in df.columns:
        if column == config.target or pd.api.types.is_numeric_dtype(df[column]):
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().mean() > 0.95:
            LOGGER.info("Coerced '%s' from text to numeric", column)
            df[column] = converted

    return df.reset_index(drop=True)


def split_columns(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Partition predictors into numeric and categorical columns."""
    predictors = [column for column in frame.columns if column != target]
    numeric = [
        column for column in predictors
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical = [column for column in predictors if column not in numeric]
    return numeric, categorical


def drop_uninformative(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Remove near-zero-variance and strongly correlated predictors.

    Selection is computed on the training partition only and then applied
    verbatim to the test partition and to anything scored later, so the decision
    is part of the model rather than of the data.
    """
    dropped: dict[str, list[str]] = {"near_zero_variance": [], "correlated": []}

    constant_numeric = [
        column for column in numeric
        if float(np.nanvar(frame[column].astype(float).to_numpy())) <= NEAR_ZERO_VARIANCE
    ]
    constant_categorical = [
        column for column in categorical if frame[column].nunique(dropna=False) <= 1
    ]
    dropped["near_zero_variance"] = constant_numeric + constant_categorical
    numeric = [column for column in numeric if column not in constant_numeric]
    categorical = [column for column in categorical if column not in constant_categorical]

    if len(numeric) > 1:
        correlations = frame[numeric].corr(numeric_only=True).abs()
        upper = correlations.where(
            np.triu(np.ones(correlations.shape, dtype=bool), k=1)
        )
        redundant: list[str] = []
        for column in upper.columns:
            partners = upper[column].dropna()
            if (partners >= correlation_threshold).any() and column not in redundant:
                # Keep the earlier column of the pair, drop the later one.
                redundant.append(column)
        dropped["correlated"] = redundant
        numeric = [column for column in numeric if column not in redundant]

    for reason, columns in dropped.items():
        if columns:
            LOGGER.info("Dropped (%s): %s", reason.replace("_", " "), ", ".join(columns))
    return numeric, categorical, dropped


def prepare(
    config: DatasetConfig,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    max_rows: int | None = None,
) -> PreparedData:
    """Run the full data layer for one dataset configuration.

    ``max_rows`` draws a stratified subsample before the split. It exists so that
    a 165,000-row dataset can be iterated on quickly; it is not the default,
    because throwing away data to save minutes is a poor trade for a final model.
    """
    snapshot = dataset_fingerprint(config.csv_path)
    frame = clean(load_csv(config.csv_path), config)

    if max_rows and max_rows < len(frame):
        frame, _ = train_test_split(
            frame,
            train_size=max_rows,
            random_state=seed,
            stratify=frame[config.target],
        )
        frame = frame.reset_index(drop=True)
        LOGGER.warning("Subsampled to %d rows (--max-rows)", len(frame))
        snapshot["subsampled_to"] = int(max_rows)

    frame = feature_layer.engineer_features(frame)

    y = frame[config.target]
    if config.task == "binary" and not pd.api.types.is_numeric_dtype(y):
        y = (y.astype(str) == str(config.positive_label)).astype(int)
    X = frame.drop(columns=[config.target])

    classes = np.array(sorted(pd.unique(y)))
    if config.task == "binary" and len(classes) != 2:
        raise ValueError(
            f"binary task declared but target '{config.target}' has "
            f"{len(classes)} distinct values"
        )
    if config.class_order:
        ordered = [c for c in config.class_order if c in set(classes.tolist())]
        if len(ordered) == len(classes):
            classes = np.array(ordered)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    numeric, categorical = split_columns(X_train, target=config.target)
    numeric, categorical, dropped = drop_uninformative(X_train, numeric, categorical)
    keep = numeric + categorical

    distribution = y_train.value_counts(normalize=True).sort_index().to_dict()
    LOGGER.info(
        "Split: %d train / %d test rows | %d numeric + %d categorical predictors",
        len(X_train), len(X_test), len(numeric), len(categorical),
    )
    LOGGER.info(
        "Training class distribution: %s",
        ", ".join(f"{label}={share:.1%}" for label, share in distribution.items()),
    )
    LOGGER.info("Feature categories: %s", feature_layer.category_summary(keep))

    snapshot.update(
        {
            "rows_total": int(len(frame)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "class_distribution_train": {str(k): float(v) for k, v in distribution.items()},
        }
    )

    return PreparedData(
        X_train=X_train[keep].reset_index(drop=True),
        X_test=X_test[keep].reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        numeric_columns=numeric,
        categorical_columns=categorical,
        classes=classes,
        dropped_columns=dropped,
        snapshot=snapshot,
    )


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute and min-max scale numerics; one-hot encode categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_sampler(y, seed: int = RANDOM_SEED):
    """Return a SMOTE sampler sized for the rarest class, or ``"passthrough"``.

    SMOTE interpolates between a minority record and its neighbours, so it needs
    at least ``k_neighbors + 1`` examples of the rarest class. On a severely
    imbalanced slice the sampler is disabled rather than allowed to fail, and the
    classifier's own class weighting carries the load.
    """
    counts = pd.Series(y).value_counts()
    minority = int(counts.min())
    if minority < 2:
        LOGGER.warning("Rarest class has %d record(s); SMOTE disabled", minority)
        return "passthrough"
    k_neighbors = max(1, min(5, minority - 1))
    return SMOTE(random_state=seed, k_neighbors=k_neighbors)


def build_pipeline(
    estimator,
    numeric: list[str],
    categorical: list[str],
    y_train=None,
    seed: int = RANDOM_SEED,
    resample: bool = True,
) -> ImbPipeline:
    """Assemble preprocessing, variance filtering, SMOTE and the classifier.

    An imbalanced-learn pipeline applies the sampler during ``fit`` only, so
    ``predict`` and ``predict_proba`` see the untouched feature space, and any
    cross-validation performed over this object resamples inside the fold.
    """
    steps = [
        (PREPROCESS_STEP, build_preprocessor(numeric, categorical)),
        (VARIANCE_STEP, VarianceThreshold(threshold=NEAR_ZERO_VARIANCE)),
    ]
    if resample and y_train is not None:
        steps.append((SAMPLER_STEP, make_sampler(y_train, seed)))
    steps.append((CLASSIFIER_STEP, estimator))
    return ImbPipeline(steps=steps)


def transformed_feature_names(pipeline: ImbPipeline) -> list[str]:
    """Names of the columns the classifier actually sees, in order."""
    preprocessor = pipeline.named_steps[PREPROCESS_STEP]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    variance = pipeline.named_steps.get(VARIANCE_STEP)
    if variance is not None:
        names = names[variance.get_support()]
    return [str(name) for name in names]


def transform_features(pipeline: ImbPipeline, X: pd.DataFrame) -> np.ndarray:
    """Push raw rows through preprocessing only, stopping before the classifier.

    This is the matrix the explanation layer works in: the classifier's own input
    space, with the sampler bypassed exactly as it is at prediction time.
    """
    matrix = pipeline.named_steps[PREPROCESS_STEP].transform(X)
    variance = pipeline.named_steps.get(VARIANCE_STEP)
    if variance is not None:
        matrix = variance.transform(matrix)
    return np.asarray(matrix, dtype=float)


def align_columns(frame: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Coerce an arbitrary uploaded frame onto the schema the model expects.

    Missing predictors are added as NaN and imputed downstream by the pipeline;
    unexpected columns are ignored. This is what allows a CRM user to upload a
    CSV that carries extra operational columns without the request failing.
    """
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = feature_layer.engineer_features(df)
    for column in expected:
        if column not in df.columns:
            df[column] = np.nan
    return df[expected]
