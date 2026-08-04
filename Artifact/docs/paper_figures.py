"""Build the figures for the paper from the registered model artefacts.

Nothing here re-trains and nothing here invents a number. The stored artefact
carries the fitted pipeline, the isotonic calibrators and the seed; the data
layer is deterministic given that seed, so the test partition reproduces
exactly and every curve drawn below is the curve the evaluation reported.
Figures that only need the summary statistics -- the joint assessment scatter,
the Shapley ranking, the fairness audit -- read them from ``reports/``.

    python -m docs.paper_figures        (from the repository root)

Output lands in ``docs/paper_figures/`` at IEEE single-column width.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from ml import calibration as calib
from ml import data_prep
from ml.config import get_dataset

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
OUT = Path(__file__).resolve().parent / "paper_figures"

# --------------------------------------------------------------------------- #
# Palette and chrome
#
# The validated default categorical palette, in slot order, on a white print
# surface. Slots 1-3 clear every all-pairs gate (CVD dE 9.2, normal-vision 24.0);
# aqua sits below 3:1 against the surface, so every chart that uses it carries a
# legend and its numbers also appear in a table in the paper. Line style doubles
# the hue on every multi-series plot, which keeps the figures readable in
# greyscale print.
# --------------------------------------------------------------------------- #

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

COL_W = 3.4  # inches: IEEE single column

plt.rcParams.update({
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECOND,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECOND,
    "ytick.labelcolor": SECOND,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def tidy(ax, *, grid: str = "y") -> None:
    """Hairline grid, no top or right rule, ticks outside."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, linestyle="-", linewidth=0.6, color=GRID, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
MODEL_COLOURS = {"logistic_regression": S1, "random_forest": S2, "xgboost": S3}
MODEL_STYLES = {"logistic_regression": "-", "random_forest": "--", "xgboost": "-."}
MODEL_MARKERS = {"logistic_regression": "o", "random_forest": "s", "xgboost": "^"}


def comparison(dataset: str) -> dict:
    path = sorted(REPORTS.glob(f"comparison__{dataset}__*.json"))[-1]
    return json.loads(path.read_text(encoding="utf-8"))


def report(model_key: str, dataset: str) -> dict:
    path = sorted(REPORTS.glob(f"{model_key}__{dataset}__*.json"))[-1]
    return json.loads(path.read_text(encoding="utf-8"))


def artefact(model_key: str, dataset: str) -> dict:
    path = sorted(MODELS.glob(f"{model_key}__{dataset}__*.joblib"))[-1]
    return joblib.load(path)


_SPLITS: dict[str, object] = {}


def test_partition(dataset: str, art: dict):
    """Reproduce the held-out partition the artefact was evaluated on."""
    key = f"{dataset}:{art['seed']}:{art['training_snapshot'].get('subsampled_to')}"
    if key not in _SPLITS:
        _SPLITS[key] = data_prep.prepare(
            get_dataset(dataset),
            seed=art["seed"],
            max_rows=art["training_snapshot"].get("subsampled_to"),
        )
    return _SPLITS[key]


def probabilities(art: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (uncalibrated, calibrated) probabilities of the positive class."""
    aligned = data_prep.align_columns(X, art["feature_columns"])
    raw = art["pipeline"].predict_proba(aligned)
    cal = calib.apply_calibration(raw, art["calibrators"])
    index = art["focus_index"]
    return raw[:, index], cal[:, index]


# --------------------------------------------------------------------------- #
# Fig. 2  Discrimination and calibration on the banking test partition
# --------------------------------------------------------------------------- #


def fig_roc_and_calibration(dataset: str = "bank_churn", name: str = "fig02-roc-calibration") -> None:
    fig, (ax_roc, ax_cal) = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.55))

    for model_key in MODEL_LABELS:
        art = artefact(model_key, dataset)
        prep = test_partition(dataset, art)
        _, p = probabilities(art, prep.X_test)
        y = prep.y_test.values
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(
            fpr, tpr,
            color=MODEL_COLOURS[model_key], linestyle=MODEL_STYLES[model_key],
            linewidth=1.4, zorder=3,
            label=f"{MODEL_LABELS[model_key]}  {roc_auc_score(y, p):.3f}",
        )
    ax_roc.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=1)
    ax_roc.text(0.62, 0.53, "chance", color=MUTED, fontsize=6, rotation=33,
                rotation_mode="anchor", ha="left", va="bottom")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_title("(a)  ROC, area under the curve in the legend", loc="left", color=INK)
    ax_roc.legend(loc="lower right", handlelength=2.2, labelspacing=0.35)
    tidy(ax_roc, grid="both")

    # Calibration: the decision layer consumes probabilities, so the gap between
    # the raw ensemble output and the isotonic-calibrated one is the part of the
    # figure that matters operationally.
    art = artefact("xgboost", dataset)
    prep = test_partition(dataset, art)
    raw, cal = probabilities(art, prep.X_test)
    y = prep.y_test.values

    ax_cal.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=1)
    ax_cal.text(0.70, 0.62, "perfect calibration", color=MUTED, fontsize=6,
                rotation=33, rotation_mode="anchor", ha="left", va="bottom")
    for probs, colour, style, marker, label in (
        (raw, S2, "--", "s", "Uncalibrated"),
        (cal, S1, "-", "o", "Isotonic"),
    ):
        obs, pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax_cal.plot(pred, obs, color=colour, linestyle=style, linewidth=1.4,
                    marker=marker, markersize=3.2, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3,
                    label=f"{label}  Brier {brier_score_loss(y, probs):.3f}")
    ax_cal.set_xlabel("Predicted probability of churn")
    ax_cal.set_ylabel("Observed churn rate")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_title("(b)  Reliability, XGBoost", loc="left", color=INK)
    ax_cal.legend(loc="upper left", handlelength=2.2, labelspacing=0.35)
    tidy(ax_cal, grid="both")

    fig.tight_layout(h_pad=1.6)
    save(fig, name)


# --------------------------------------------------------------------------- #
# Fig. 3  The two assessment axes, one panel per domain
# --------------------------------------------------------------------------- #


def fig_joint_assessment(name: str = "fig03-joint-assessment") -> None:
    panels = [("bank_churn", "(a)  Retail banking"), ("netflix_churn", "(b)  Streaming")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, COL_W * 0.62), sharey=True)

    for ax, (dataset, title) in zip(axes, panels):
        for row in comparison(dataset)["models"]:
            key = row["model_key"]
            ax.scatter(
                row["mcc"], row["shap_lime_overlap"],
                s=34, color=MODEL_COLOURS[key], marker=MODEL_MARKERS[key],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("MCC")
        tidy(ax, grid="both")

    axes[0].set_ylabel("SHAP-LIME top-5 overlap")
    axes[0].set_ylim(0.60, 0.90)
    axes[0].set_xlim(0.42, 0.62)
    axes[0].set_xticks([0.45, 0.50, 0.55, 0.60])
    axes[1].set_xlim(0.74, 1.02)
    axes[1].set_xticks([0.80, 0.90, 1.00])

    handles = [
        Line2D([], [], linestyle="none", marker=MODEL_MARKERS[k], color=MODEL_COLOURS[k],
               markeredgecolor="white", markeredgewidth=0.8, markersize=5, label=MODEL_LABELS[k])
        for k in MODEL_LABELS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, columnspacing=1.0,
               handletextpad=0.3, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(w_pad=1.2)
    save(fig, name)


# --------------------------------------------------------------------------- #
# Fig. 4  What the recommended banking model reads, and what it may act on
# --------------------------------------------------------------------------- #


def fig_global_importance(
    model_key: str = "xgboost",
    dataset: str = "bank_churn",
    top_n: int = 12,
    name: str = "fig04-shap-importance",
) -> None:
    rows = report(model_key, dataset)["global_importance"][:top_n][::-1]
    labels = [r["label"] for r in rows]
    values = [r["mean_abs_shap"] for r in rows]
    colours = [S2 if r["protected"] else S1 for r in rows]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.92))
    ax.barh(range(len(rows)), values, height=0.68, color=colours, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean absolute Shapley value")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    tidy(ax, grid="x")

    handles = [
        Line2D([], [], marker="s", linestyle="none", color=S1, markersize=5,
               label="Actionable: may trigger an offer"),
        Line2D([], [], marker="s", linestyle="none", color=S2, markersize=5,
               label="Demographic: explained, never acted on"),
    ]
    # Above the plot: the bars run the full width, so there is no interior
    # region a two-line legend can occupy without covering a label.
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.01),
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0)
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------- #
# Fig. 5  The threshold is a business decision
# --------------------------------------------------------------------------- #


def fig_threshold_value(
    model_key: str = "xgboost",
    dataset: str = "bank_churn",
    name: str = "fig05-threshold-value",
) -> None:
    art = artefact(model_key, dataset)
    prep = test_partition(dataset, art)
    _, p = probabilities(art, prep.X_test)
    y = prep.y_test.values
    econ = art["economics"]

    def value_at(t: float) -> float:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        return (
            tp * econ["value_true_positive"]
            - fp * econ["cost_false_positive"]
            - fn * econ["cost_false_negative"]
            + tn * econ["value_true_negative"]
        ) / len(y)

    grid = np.linspace(0.02, 0.98, 193)
    value = np.array([value_at(t) for t in grid])
    chosen = art["threshold"]
    chosen_value = value_at(chosen)      # evaluated at the threshold itself, not
    default_value = value_at(0.5)        # interpolated between two grid points

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.68))
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.plot(grid, value, color=S1, linewidth=1.3, zorder=3)
    ax.scatter([chosen], [chosen_value], s=26, color=S1, edgecolor="white",
               linewidth=0.8, zorder=4)
    ax.scatter([0.5], [default_value], s=26, color=S2, marker="s",
               edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate(
        f"selected {chosen:.2f}: {chosen_value:+.2f}",
        xy=(chosen, chosen_value), xytext=(0.03, -21),
        color=SECOND, fontsize=6, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                        shrinkA=1, shrinkB=2),
    )
    ax.annotate(
        f"default 0.50: {default_value:+.2f}",
        xy=(0.5, default_value), xytext=(0.60, 6),
        color=SECOND, fontsize=6, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                        shrinkA=1, shrinkB=2),
    )
    ax.set_xlabel("Operating threshold")
    ax.set_ylabel("Campaign value per customer")
    ax.set_xlim(0, 1)
    tidy(ax, grid="both")
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------- #
# Fig. 6  Fairness audit
# --------------------------------------------------------------------------- #

AGE_ORDER = ["Under 30", "30-44", "45-59", "60 and over"]


def fig_fairness(
    model_key: str = "xgboost",
    dataset: str = "bank_churn",
    name: str = "fig06-fairness",
) -> None:
    fairness = report(model_key, dataset)["fairness"]
    attributes = list(fairness["attributes"].items())
    heights = [max(len(a[1]["segments"]), 2) for a in attributes]

    fig, axes = plt.subplots(
        len(attributes), 1,
        figsize=(COL_W, COL_W * 0.95),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )

    for ax, (key, block) in zip(axes, attributes):
        segments = list(block["segments"])
        if key.lower() == "age":
            order = {g: i for i, g in enumerate(AGE_ORDER)}
            segments.sort(key=lambda s: order.get(s["group"], 99))
        segments = segments[::-1]
        groups = [s["group"] for s in segments]
        rates = [s["selection_rate"] for s in segments]
        floor = 0.8 * max(rates)

        ax.barh(range(len(groups)), rates, height=0.62, color=S1, zorder=3)
        ax.axvline(floor, color=S2, linestyle="--", linewidth=1.0, zorder=4)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)
        ax.set_xlim(0, 0.82)
        ax.set_ylim(-0.65, len(groups) - 0.35)
        ax.set_title(block["label"], loc="left", color=INK, fontsize=7)
        for i, rate in enumerate(rates):
            ax.text(rate + 0.012, i, f"{rate:.2f}", va="center", ha="left",
                    fontsize=6, color=SECOND)
        tidy(ax, grid="x")

    first = 0.8 * max(s["selection_rate"] for s in attributes[0][1]["segments"])
    axes[0].annotate(
        "four-fifths of the highest-\nselected group: bars left of\nthis line fail the convention",
        xy=(first, 0.45), xytext=(first + 0.16, 0.45),
        color=SECOND, fontsize=6, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                        shrinkA=1, shrinkB=1),
    )
    axes[-1].set_xlabel("Share of the group flagged for a retention offer")
    fig.tight_layout(h_pad=0.9)
    save(fig, name)


# --------------------------------------------------------------------------- #
# Fig. 9  Calibration and the price of the threshold, in one column
#
# The two halves of the same argument -- the decision layer consumes
# probabilities, and where it cuts them is worth money -- drawn together so the
# eight-page version of the paper can carry both.
# --------------------------------------------------------------------------- #


def fig_calibration_and_threshold(
    model_key: str = "xgboost",
    dataset: str = "bank_churn",
    name: str = "fig09-calibration-threshold",
) -> None:
    art = artefact(model_key, dataset)
    prep = test_partition(dataset, art)
    raw, cal = probabilities(art, prep.X_test)
    y = prep.y_test.values
    econ = art["economics"]

    fig, (ax_cal, ax_val) = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.24))

    ax_cal.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=1)
    ax_cal.text(0.66, 0.585, "perfect calibration", color=MUTED, fontsize=5.8,
                rotation=33, rotation_mode="anchor", ha="left", va="bottom")
    for probs, colour, style, marker, label in (
        (raw, S2, "--", "s", "Uncalibrated"),
        (cal, S1, "-", "o", "Isotonic"),
    ):
        obs, pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax_cal.plot(pred, obs, color=colour, linestyle=style, linewidth=1.3,
                    marker=marker, markersize=2.8, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3,
                    label=f"{label}  Brier {brier_score_loss(y, probs):.3f}")
    ax_cal.set_xlabel("Predicted probability of churn")
    ax_cal.set_ylabel("Observed churn rate")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_title("(a)  Reliability", loc="left", color=INK)
    ax_cal.legend(loc="upper left", handlelength=2.0, labelspacing=0.3)
    tidy(ax_cal, grid="both")

    def value_at(t: float) -> float:
        pred = (cal >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        return (
            tp * econ["value_true_positive"]
            - fp * econ["cost_false_positive"]
            - fn * econ["cost_false_negative"]
            + tn * econ["value_true_negative"]
        ) / len(y)

    grid = np.linspace(0.02, 0.98, 193)
    value = np.array([value_at(t) for t in grid])
    chosen = art["threshold"]

    ax_val.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax_val.plot(grid, value, color=S1, linewidth=1.3, zorder=3)
    ax_val.scatter([chosen], [value_at(chosen)], s=24, color=S1,
                   edgecolor="white", linewidth=0.8, zorder=4)
    ax_val.scatter([0.5], [value_at(0.5)], s=24, color=S2, marker="s",
                   edgecolor="white", linewidth=0.8, zorder=4)
    ax_val.annotate(f"selected {chosen:.2f}: {value_at(chosen):+.2f}",
                    xy=(chosen, value_at(chosen)), xytext=(0.03, -22),
                    color=SECOND, fontsize=5.8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                    shrinkA=1, shrinkB=2))
    ax_val.annotate(f"default 0.50: {value_at(0.5):+.2f}",
                    xy=(0.5, value_at(0.5)), xytext=(0.60, 6),
                    color=SECOND, fontsize=5.8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                    shrinkA=1, shrinkB=2))
    ax_val.set_xlabel("Operating threshold")
    ax_val.set_ylabel("Campaign value per customer")
    ax_val.set_xlim(0, 1)
    ax_val.set_title("(b)  What the threshold is worth", loc="left", color=INK)
    tidy(ax_val, grid="both")

    fig.tight_layout(h_pad=1.4)
    save(fig, name)


def main() -> None:
    print("building paper figures")
    fig_roc_and_calibration()
    fig_joint_assessment()
    fig_global_importance()
    fig_threshold_value()
    fig_fairness()
    fig_calibration_and_threshold()
    print("done")


if __name__ == "__main__":
    main()
