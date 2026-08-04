"""Explanation layer: Shapley attributions, audited against a local surrogate.

SHAP expresses every prediction as an additive attribution over the features,
g(z') = phi_0 + sum_j phi_j z'_j, where phi_0 is the base value (the mean model
output) and phi_j is the Shapley value of feature j. Tree ensembles are explained
exactly and in polynomial time with TreeExplainer; logistic regression is
explained with the linear explainer.

Because post-hoc explanations can be fragile, SHAP is audited against LIME on a
fixed sample of test-set predictions stratified across the risk spectrum. For
each sampled record the top five features by absolute attribution are extracted
from both methods and compared by rank agreement. High agreement raises
confidence that an explanation reflects the model rather than the idiosyncrasies
of one explainer; systematic disagreement localises the features whose
attributions should not be shown to business users without caveats.

Two rules discipline how the output is read, and they are enforced in the
dashboard copy as well as here:

1. an attribution describes the behaviour of the model, not the causal structure
   of the world;
2. any driver whose direction contradicts domain knowledge is escalated for
   data-quality review before the model is trusted.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np

from ml import features as feature_layer
from ml.config import (
    EXPLANATION_AUDIT_SAMPLE,
    RANDOM_SEED,
    SHAP_BACKGROUND_SIZE,
    TOP_DRIVERS,
    get_logger,
)

LOGGER = get_logger(__name__)

TREE_MODELS = {"random_forest", "xgboost"}


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


def background_sample(
    matrix: np.ndarray,
    size: int = SHAP_BACKGROUND_SIZE,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Draw the reference distribution stored alongside the model artefact."""
    matrix = np.asarray(matrix, dtype=float)
    if len(matrix) <= size:
        return matrix.copy()
    rng = np.random.default_rng(seed)
    return matrix[rng.choice(len(matrix), size=size, replace=False)].copy()


def build_explainer(estimator, background: np.ndarray, model_key: str):
    """Return a SHAP explainer appropriate to the estimator family.

    TreeExplainer is used for the ensembles because it is exact and cheap enough
    to run at scoring time, which is what keeps the explanation layer
    computationally honest rather than aspirational.
    """
    import shap

    background = np.asarray(background, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_key in TREE_MODELS or hasattr(estimator, "estimators_") or hasattr(
            estimator, "get_booster"
        ):
            return shap.TreeExplainer(estimator)
        if hasattr(estimator, "coef_"):
            return shap.LinearExplainer(estimator, background)
        # Last resort for an estimator family added later: model-agnostic,
        # slower, but never wrong about which model it is describing.
        summary = shap.kmeans(background, min(25, len(background)))
        return shap.KernelExplainer(estimator.predict_proba, summary)


def shap_matrix(explainer, matrix: np.ndarray, class_index: int, n_classes: int) -> np.ndarray:
    """Return the (rows x features) attribution matrix for one class.

    SHAP returns different shapes for different model families and versions --
    ``(n, f)``, ``(n, f, k)``, ``(k, n, f)`` or a list of ``(n, f)`` arrays --
    so the shape is normalised here once and never thought about again.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows, n_features = matrix.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            values = np.asarray(explainer(matrix).values, dtype=float)
        except Exception:  # noqa: BLE001 - explainer APIs vary across versions
            values = explainer.shap_values(matrix)
            if isinstance(values, list):
                values = np.asarray(values, dtype=float)
            values = np.asarray(values, dtype=float)

    if values.ndim == 2:
        # Binary tree/linear models emit attributions for the positive class
        # only; the negative class is their mirror image.
        return values if class_index == 1 or n_classes <= 2 else -values
    if values.ndim == 3:
        if values.shape[0] == rows and values.shape[1] == n_features:
            return values[:, :, min(class_index, values.shape[2] - 1)]
        if values.shape[1] == rows and values.shape[2] == n_features:
            return values[min(class_index, values.shape[0] - 1), :, :]
    raise ValueError(f"unexpected SHAP output shape {values.shape}")


def base_value(explainer, class_index: int) -> float:
    """The expected model output, phi_0 in the additive decomposition."""
    expected = getattr(explainer, "expected_value", 0.0)
    expected = np.asarray(expected, dtype=float).ravel()
    if expected.size == 0:
        return 0.0
    return float(expected[min(class_index, expected.size - 1)])


def global_importance(
    attributions: np.ndarray,
    feature_names: Sequence[str],
    top_k: int | None = None,
) -> list[dict]:
    """Mean absolute Shapley value per feature: which drivers dominate overall.

    This is the natural input to strategic decisions -- a pricing review, a
    supplier renegotiation -- as distinct from the per-record view that the
    decision layer consumes.
    """
    mean_abs = np.abs(np.asarray(attributions, dtype=float)).mean(axis=0)
    mean_signed = np.asarray(attributions, dtype=float).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    if top_k:
        order = order[:top_k]

    ranked = []
    for position, index in enumerate(order, start=1):
        name = str(feature_names[index])
        described = feature_layer.describe(name, feature_names)
        ranked.append(
            {
                "rank": position,
                "feature": name,
                "label": described["label"],
                "category": described["category"],
                "category_label": described["category_label"],
                "protected": described["protected"],
                "mean_abs_shap": float(mean_abs[index]),
                "mean_shap": float(mean_signed[index]),
                "direction": "increases" if mean_signed[index] >= 0 else "decreases",
            }
        )
    return ranked


def local_drivers(
    attribution_row: np.ndarray,
    values_row: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = TOP_DRIVERS,
) -> list[dict]:
    """Decompose one record's estimate into its strongest named contributions."""
    attribution_row = np.asarray(attribution_row, dtype=float).ravel()
    values_row = np.asarray(values_row, dtype=float).ravel()
    order = np.argsort(np.abs(attribution_row))[::-1][:top_k]

    drivers = []
    for position, index in enumerate(order, start=1):
        name = str(feature_names[index])
        described = feature_layer.describe(name, feature_names)
        contribution = float(attribution_row[index])
        drivers.append(
            {
                "rank": position,
                "feature": name,
                "column": described["column"],
                "label": described["label"],
                "category": described["category"],
                "category_label": described["category_label"],
                "protected": described["protected"],
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "direction": "increases" if contribution >= 0 else "decreases",
                "scaled_value": float(values_row[index]) if index < values_row.size else None,
            }
        )
    return drivers


# --------------------------------------------------------------------------- #
# LIME cross-validation of the explanations
# --------------------------------------------------------------------------- #


def stratified_audit_sample(
    proba_positive: np.ndarray,
    size: int = EXPLANATION_AUDIT_SAMPLE,
    bins: int = 5,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Sample record indices evenly across the risk spectrum, not at random.

    Auditing only the confident predictions would flatter the explainers; the
    sample therefore spans the probability range in equal-width bands.
    """
    proba_positive = np.asarray(proba_positive, dtype=float)
    rows = len(proba_positive)
    if rows <= size:
        return np.arange(rows)

    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, bins + 1)
    per_bin = max(1, size // bins)
    chosen: list[int] = []
    for low, high in zip(edges[:-1], edges[1:]):
        members = np.flatnonzero((proba_positive >= low) & (proba_positive < high))
        if members.size:
            take = min(per_bin, members.size)
            chosen.extend(rng.choice(members, size=take, replace=False).tolist())

    remaining = size - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(rows), np.asarray(chosen, dtype=int))
        if pool.size:
            chosen.extend(
                rng.choice(pool, size=min(remaining, pool.size), replace=False).tolist()
            )
    return np.asarray(sorted(chosen[:size]), dtype=int)


def _top_indices(weights: dict[int, float], k: int) -> list[int]:
    return [
        index
        for index, _ in sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
    ]


def _rank_agreement(first: Sequence[int], second: Sequence[int]) -> dict[str, float]:
    """Overlap and ordering agreement between two ranked feature lists."""
    set_first, set_second = set(first), set(second)
    union = set_first | set_second
    intersection = set_first & set_second
    jaccard = len(intersection) / len(union) if union else 0.0
    overlap = len(intersection) / max(len(set_first), 1)

    # Spearman footrule over the shared features, normalised to [0, 1].
    if len(intersection) >= 2:
        rank_first = {feature: position for position, feature in enumerate(first)}
        rank_second = {feature: position for position, feature in enumerate(second)}
        distance = sum(abs(rank_first[f] - rank_second[f]) for f in intersection)
        worst = (len(intersection) ** 2) / 2 or 1
        footrule = max(0.0, 1.0 - distance / worst)
    else:
        footrule = float(len(intersection))
    return {"jaccard": jaccard, "overlap": overlap, "order_agreement": footrule}


def lime_audit(
    estimator,
    background: np.ndarray,
    audit_matrix: np.ndarray,
    shap_attributions: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    class_index: int,
    top_k: int = 5,
    num_samples: int = 1500,
    seed: int = RANDOM_SEED,
    predict_proba: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Compare SHAP and LIME top-k attributions, and measure LIME's own stability.

    Returns the mean top-k overlap between the two explainers, the mean ordering
    agreement, the per-feature disagreement counts (which localise the features
    to caveat), and the stability of LIME across two independently seeded runs.
    """
    from lime.lime_tabular import LimeTabularExplainer

    background = np.asarray(background, dtype=float)
    audit_matrix = np.asarray(audit_matrix, dtype=float)
    shap_attributions = np.asarray(shap_attributions, dtype=float)
    predict_proba = predict_proba or estimator.predict_proba

    LOGGER.info("Auditing SHAP against LIME on %d records", len(audit_matrix))

    def make_explainer(random_state: int) -> LimeTabularExplainer:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LimeTabularExplainer(
                training_data=background,
                feature_names=[str(name) for name in feature_names],
                class_names=[str(name) for name in class_names],
                mode="classification",
                discretize_continuous=True,
                random_state=random_state,
            )

    primary = make_explainer(seed)
    secondary = make_explainer(seed + 1)

    agreements: list[dict[str, float]] = []
    stabilities: list[float] = []
    disagreement_counts: dict[str, int] = {}
    shap_top_counts: dict[str, int] = {}

    for position, row in enumerate(audit_matrix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanation = primary.explain_instance(
                row, predict_proba, num_features=top_k, labels=(class_index,),
                num_samples=num_samples,
            )
            repeat = secondary.explain_instance(
                row, predict_proba, num_features=top_k, labels=(class_index,),
                num_samples=num_samples,
            )

        lime_top = _top_indices(dict(explanation.as_map()[class_index]), top_k)
        lime_repeat = _top_indices(dict(repeat.as_map()[class_index]), top_k)
        shap_row = shap_attributions[position]
        shap_top = list(np.argsort(np.abs(shap_row))[::-1][:top_k])

        agreements.append(_rank_agreement(shap_top, lime_top))
        stabilities.append(_rank_agreement(lime_top, lime_repeat)["jaccard"])

        for index in shap_top:
            name = str(feature_names[index])
            shap_top_counts[name] = shap_top_counts.get(name, 0) + 1
            if index not in lime_top:
                disagreement_counts[name] = disagreement_counts.get(name, 0) + 1

    def mean(key: str) -> float:
        return float(np.mean([entry[key] for entry in agreements])) if agreements else 0.0

    # A feature that SHAP ranks highly but LIME rarely corroborates is exactly
    # the one to caveat in the interface.
    contested = sorted(
        (
            {
                "feature": name,
                "label": feature_layer.friendly_name(name, feature_names),
                "shap_top_k_count": shap_top_counts.get(name, 0),
                "lime_disagreements": count,
                "disagreement_rate": count / max(shap_top_counts.get(name, 1), 1),
            }
            for name, count in disagreement_counts.items()
            if shap_top_counts.get(name, 0) >= 5
        ),
        key=lambda entry: entry["disagreement_rate"],
        reverse=True,
    )[:10]

    audit = {
        "records_audited": int(len(audit_matrix)),
        "top_k": top_k,
        "mean_top_k_overlap": mean("overlap"),
        "mean_jaccard": mean("jaccard"),
        "mean_order_agreement": mean("order_agreement"),
        "lime_stability": float(np.mean(stabilities)) if stabilities else 0.0,
        "contested_features": contested,
    }
    LOGGER.info(
        "Explanation audit: top-%d overlap %.1f%% | ordering %.1f%% | LIME stability %.1f%%",
        top_k, 100 * audit["mean_top_k_overlap"],
        100 * audit["mean_order_agreement"], 100 * audit["lime_stability"],
    )
    if audit["mean_top_k_overlap"] < 0.5:
        LOGGER.warning(
            "SHAP and LIME agree on fewer than half of the leading drivers; "
            "treat per-record explanations as provisional until investigated."
        )
    return audit
