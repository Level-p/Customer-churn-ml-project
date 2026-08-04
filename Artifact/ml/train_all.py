"""Train the whole ladder and compare the models on both axes.

The comparison is deliberately not a single leaderboard. A model is assessed on
discrimination *and* calibration *and* the stability and fidelity of the
explanations it supports, because a model whose explanations cannot be trusted
cannot drive a decision layer, however well it ranks records.

    python -m ml.train_all
    python -m ml.train_all --dataset retail_demand_bands --trials 40
    python -m ml.train_all --fast          # smoke test in a couple of minutes

The selection rule at the end is a recommendation, not an executive decision. It
is stated in the output so a reader can disagree with it on the evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

from ml import train_logistic_regression as lr
from ml import train_random_forest as rf
from ml import train_xgboost as xgb
from ml.config import REPORTS_DIR, get_logger
from ml.train_common import TrainingOutcome, common_parser, train

LOGGER = get_logger("ml.train_all")

SPECS = {
    lr.MODEL_KEY: {"build": lr.build_estimator, "tune": lr.tune, "finalise": None,
                   "after": lr.log_coefficients},
    rf.MODEL_KEY: {"build": rf.build_estimator, "tune": rf.tune, "finalise": None,
                   "after": rf.log_impurity_importance},
    xgb.MODEL_KEY: {"build": xgb.build_estimator, "tune": xgb.tune, "finalise": xgb.finalise,
                    "after": xgb.log_feature_importance},
}


def comparison_rows(outcomes: dict[str, TrainingOutcome]) -> list[dict]:
    rows = []
    for key, outcome in outcomes.items():
        bundle = outcome.bundle
        metrics = bundle["metrics"]
        audit = bundle.get("explanation_audit") or {}
        profit = (bundle["evaluation"].get("profit") or {})
        rows.append(
            {
                "model": bundle["model_label"],
                "model_key": key,
                "version": bundle["version"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "mcc": metrics["mcc"],
                "brier": metrics["brier"],
                "shap_lime_overlap": audit.get("mean_top_k_overlap"),
                "lime_stability": audit.get("lime_stability"),
                "value_per_record": profit.get("value_per_record"),
                "threshold": bundle["threshold"],
                "training_seconds": bundle["training_seconds"],
            }
        )
    return rows


def _format(value, spec: str = "{:.4f}", width: int = 9) -> str:
    if value is None:
        return "n/a".rjust(width)
    return spec.format(value).rjust(width)


def log_comparison(rows: list[dict]) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 118)
    LOGGER.info("MODEL COMPARISON  |  discrimination and calibration  |  explanation quality")
    LOGGER.info("=" * 118)
    LOGGER.info(
        "%-22s %9s %9s %9s %9s %9s %9s | %11s %11s | %9s",
        "model", "precision", "recall", "F1", "ROC-AUC", "MCC", "Brier",
        "SHAP~LIME", "LIME stab.", "value/rec",
    )
    LOGGER.info("-" * 118)
    for row in rows:
        LOGGER.info(
            "%-22s %s %s %s %s %s %s | %s %s | %s",
            row["model"][:22],
            _format(row["precision"]), _format(row["recall"]), _format(row["f1"]),
            _format(row["roc_auc"]), _format(row["mcc"]), _format(row["brier"]),
            _format(row["shap_lime_overlap"], "{:.1%}", 11),
            _format(row["lime_stability"], "{:.1%}", 11),
            _format(row["value_per_record"], "{:.2f}"),
        )
    LOGGER.info("-" * 118)
    LOGGER.info(
        "Brier is a loss: lower is better. SHAP~LIME is the mean top-5 overlap between the two "
        "explainers; LIME stab. is its agreement with itself across two seeds."
    )


def recommend(rows: list[dict]) -> dict:
    """Recommend a model on the joint profile, and say why.

    The rule is deliberately blunt and deliberately conservative:

    * rank the candidates by MCC, which is the least forgiving summary of a
      confusion matrix under imbalance;
    * disqualify any model whose explanations failed the audit, because the
      decision layer cannot be fed by an explainer the audit did not trust;
    * if the winning ensemble beats the transparent baseline by less than one
      point of MCC, recommend the baseline instead. An unexplainable model has to
      earn its opacity, and a hair's-breadth win does not.
    """
    baseline = next((row for row in rows if row["model_key"] == "logistic_regression"), None)
    audited = [
        row for row in rows
        if row["shap_lime_overlap"] is None or row["shap_lime_overlap"] >= 0.5
    ]
    disqualified = [row["model"] for row in rows if row not in audited]
    pool = audited or rows

    best = max(pool, key=lambda row: row["mcc"])
    reasons = [f"highest MCC among audited models ({best['mcc']:.4f})"]

    if baseline and best["model_key"] != "logistic_regression":
        margin = best["mcc"] - baseline["mcc"]
        if margin < 0.01:
            best = baseline
            reasons = [
                f"the leading ensemble improves MCC by only {margin:.4f} over the "
                "transparent baseline, which does not justify the loss of direct "
                "interpretability"
            ]
        else:
            reasons.append(f"beats the logistic baseline by {margin:.4f} MCC")

    if disqualified:
        reasons.append(
            "excluded for failing the explanation audit: " + ", ".join(disqualified)
        )

    LOGGER.info("")
    LOGGER.info("RECOMMENDED FOR DEPLOYMENT: %s (version %s)", best["model"], best["version"])
    for reason in reasons:
        LOGGER.info("  - %s", reason)
    LOGGER.info(
        "  - the dashboard can still serve any registered model; this is the default, "
        "not a lock-in"
    )
    return {"model_key": best["model_key"], "version": best["version"], "reasons": reasons}


def main() -> None:
    parser = common_parser("Train and compare every model in the BDSS ladder.")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials for XGBoost")
    parser.add_argument(
        "--models", nargs="*", default=list(SPECS), choices=list(SPECS),
        help="subset of the ladder to train",
    )
    args = parser.parse_args()

    outcomes: dict[str, TrainingOutcome] = {}
    for key in args.models:
        spec = SPECS[key]
        outcome = train(
            model_key=key,
            build_estimator=spec["build"],
            tune=spec["tune"],
            args=copy.copy(args),
            finalise=spec["finalise"],
        )
        spec["after"](outcome)
        outcomes[key] = outcome

    rows = comparison_rows(outcomes)
    log_comparison(rows)
    recommendation = recommend(rows)

    dataset = next(iter(outcomes.values())).bundle["dataset"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = REPORTS_DIR / f"comparison__{dataset}__{stamp}.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset,
                "generated_at": stamp,
                "seed": args.seed,
                "models": rows,
                "recommendation": recommendation,
            },
            handle,
            indent=2,
        )
    LOGGER.info("")
    LOGGER.info("Comparison written to %s", destination)


if __name__ == "__main__":
    main()
