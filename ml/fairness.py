"""Fairness audit.

A churn model trained on historical data can encode historical inequity, and
features such as geography or salary can proxy for protected attributes. A
retention system makes that concrete in a way a pure classifier does not: if
offers flow only to the customers a model flags, the groups it systematically
fails to flag quietly receive worse treatment, and nobody ever sees a complaint
about an offer that was never made.

The audit therefore reports, for each declared sensitive attribute:

* the selection rate per group -- what share of the group the model would target;
* the true positive rate per group -- what share of that group's actual churners
  the model catches, which is the group's share of the *protection* on offer;
* the false positive rate and precision per group;
* the ratio of the lowest to the highest selection rate, compared against the
  four-fifths convention;
* whether any demographic predictor is behaving as a leading driver, which is
  the signal to investigate it as a proxy.

None of these numbers decides anything on its own. A selection-rate gap can be a
faithful reflection of a real difference in churn between groups, and equalising
it by fiat would deny help to the group that needs it. What the audit does is
make the gap impossible to ship without having looked at it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import get_logger
from ml.features import PROTECTED_CATEGORIES, categorise, friendly_name

LOGGER = get_logger(__name__)

#: The four-fifths convention: a selection-rate ratio below this is the
#: conventional trigger for investigation, not a verdict.
SELECTION_RATIO_FLOOR = 0.80

#: A gap in the true positive rate above this means one group's churners are
#: materially less likely to be offered anything.
TPR_GAP_CEILING = 0.10

#: Groups smaller than this are reported but not flagged: the rates are noise.
MIN_SEGMENT = 30

AGE_BINS = [0, 30, 45, 60, 200]
AGE_LABELS = ["Under 30", "30-44", "45-59", "60 and over"]


def _segment(series: pd.Series) -> pd.Series:
    """Group a sensitive attribute into segments; bin ages into bands."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        return pd.cut(series, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return series.astype("string").fillna("Missing")


def audit(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attributes: tuple[str, ...],
    positive_label=1,
    global_importance: list[dict] | None = None,
    top_k: int = 10,
) -> dict:
    """Compare error rates across the groups of each sensitive attribute."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results: dict[str, dict] = {}
    flags: list[str] = []

    for attribute in sensitive_attributes:
        if attribute not in frame.columns:
            LOGGER.info("Fairness: '%s' is not in the data; skipped", attribute)
            continue

        segments = _segment(frame[attribute])
        rows: list[dict] = []
        for group in [g for g in pd.unique(segments) if pd.notna(g)]:
            mask = (segments == group).to_numpy()
            size = int(mask.sum())
            if size == 0:
                continue

            truth = y_true[mask] == positive_label
            flagged = y_pred[mask] == positive_label
            true_positives = int(np.sum(truth & flagged))
            actual_positives = int(np.sum(truth))
            actual_negatives = int(np.sum(~truth))
            false_positives = int(np.sum(~truth & flagged))

            rows.append(
                {
                    "group": str(group),
                    "records": size,
                    "base_rate": float(np.mean(truth)),
                    "selection_rate": float(np.mean(flagged)),
                    "true_positive_rate": (
                        true_positives / actual_positives if actual_positives else None
                    ),
                    "false_positive_rate": (
                        false_positives / actual_negatives if actual_negatives else None
                    ),
                    "precision": (
                        true_positives / int(np.sum(flagged)) if np.sum(flagged) else None
                    ),
                    "reliable": size >= MIN_SEGMENT,
                }
            )

        reliable = [row for row in rows if row["reliable"]]
        selection_rates = [row["selection_rate"] for row in reliable if row["selection_rate"] > 0]
        tprs = [row["true_positive_rate"] for row in reliable if row["true_positive_rate"] is not None]

        ratio = (min(selection_rates) / max(selection_rates)) if len(selection_rates) > 1 else None
        tpr_gap = (max(tprs) - min(tprs)) if len(tprs) > 1 else None

        attribute_flags: list[str] = []
        if ratio is not None and ratio < SELECTION_RATIO_FLOOR:
            attribute_flags.append(
                f"selection-rate ratio {ratio:.2f} is below the four-fifths convention"
            )
        if tpr_gap is not None and tpr_gap > TPR_GAP_CEILING:
            attribute_flags.append(
                f"true-positive-rate gap of {tpr_gap:.2f} between groups: the churners "
                "in one group are materially less likely to be offered anything"
            )

        results[attribute] = {
            "label": friendly_name(attribute),
            "segments": sorted(rows, key=lambda row: row["records"], reverse=True),
            "selection_rate_ratio": ratio,
            "true_positive_rate_gap": tpr_gap,
            "flags": attribute_flags,
        }
        flags.extend(f"{attribute}: {flag}" for flag in attribute_flags)

    proxies = _proxy_candidates(global_importance or [], top_k=top_k)
    summary = {
        "sensitive_attributes": list(sensitive_attributes),
        "attributes": results,
        "proxy_candidates": proxies,
        "flags": flags,
        "selection_ratio_floor": SELECTION_RATIO_FLOOR,
        "tpr_gap_ceiling": TPR_GAP_CEILING,
    }
    _log(summary)
    return summary


def _proxy_candidates(global_importance: list[dict], top_k: int) -> list[dict]:
    """Demographic predictors among the model's leading drivers.

    Appearing here is not a conviction. It means the attribute is doing real work
    in the model, and that the work needs an independent business justification;
    if it has none, the feature comes out, and if it has one, the reason is
    recorded. Either way the decision layer refuses to turn it into an offer.
    """
    leading = global_importance[:top_k]
    return [
        {
            "feature": driver["feature"],
            "label": driver["label"],
            "rank": driver["rank"],
            "mean_abs_shap": driver["mean_abs_shap"],
            "category": driver["category"],
        }
        for driver in leading
        if categorise(driver["feature"]) in PROTECTED_CATEGORIES
    ]


def _log(summary: dict) -> None:
    LOGGER.info("")
    LOGGER.info("FAIRNESS AUDIT (test partition)")
    for attribute, result in summary["attributes"].items():
        LOGGER.info("  %s", result["label"])
        LOGGER.info(
            "    %-16s %8s %10s %10s %10s %10s",
            "group", "records", "churn", "targeted", "caught", "precision",
        )
        for row in result["segments"]:
            LOGGER.info(
                "    %-16s %8d %9.1f%% %9.1f%% %9s %10s%s",
                row["group"][:16],
                row["records"],
                100 * row["base_rate"],
                100 * row["selection_rate"],
                "n/a" if row["true_positive_rate"] is None else f"{100 * row['true_positive_rate']:.1f}%",
                "n/a" if row["precision"] is None else f"{100 * row['precision']:.1f}%",
                "" if row["reliable"] else "  (small group; rates unreliable)",
            )
        for flag in result["flags"]:
            LOGGER.warning("    FLAG: %s", flag)

    if summary["proxy_candidates"]:
        LOGGER.warning(
            "  Demographic attributes among the leading drivers: %s",
            ", ".join(entry["label"] for entry in summary["proxy_candidates"]),
        )
        LOGGER.warning(
            "  Each needs an independent business justification or it comes out of the "
            "model. The decision layer will not convert one into a differential offer."
        )
    if not summary["flags"]:
        LOGGER.info("  No group-level disparity crossed a review threshold.")
