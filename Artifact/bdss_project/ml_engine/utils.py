"""Input handling, batch summaries, CSV export and the dashboard charts.

The charts are rendered server-side with matplotlib and embedded in the page as
base64 PNGs. That choice is deliberate: it keeps the dashboard free of any
external script or font request, which matters when the page is displaying
customer-level risk inside a corporate network.

Colour is used to the following rules, and only these:

* the three risk bands are a *status* scale (red / amber / green), and a status
  colour never carries meaning on its own -- every band is written out in text
  beside the mark, because red and amber are close under deuteranopia;
* the driver categories are a *categorical* scale, taken in fixed order from a
  colour-blind-validated palette, and every bar is directly labelled;
* everything else -- axes, gridlines, labels -- is recessive ink, never a series
  colour.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # no display, no GUI toolkit, no surprises under a web server

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ml.features import CATEGORIES, CATEGORY_LABELS  # noqa: E402

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

#: Status scale for the risk bands. Validated against the page surface: all three
#: clear 3:1 contrast. Their worst adjacent CVD separation is at the floor, so
#: they are only ever used with a text label alongside.
BAND_COLOURS = {"high": "#b42318", "medium": "#b54708", "low": "#067647"}

#: Priority reuses the same status scale: P1 is the thing that is on fire.
PRIORITY_COLOURS = {"P1": "#b42318", "P2": "#b54708", "P3": "#067647"}

#: Categorical scale for the feature categories, in fixed order. Assigned by
#: category, never by rank, so a filter that removes a category does not repaint
#: the survivors.
CATEGORY_COLOURS = {
    "behavioural": "#2a78d6",
    "transactional": "#1baf7a",
    "contractual": "#eda100",
    "engagement": "#008300",
    "demographic": "#4a3aa7",
    "uncategorised": "#898781",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
    }
)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


class UploadError(ValueError):
    """The uploaded file cannot be scored, with a reason a human can act on."""


def read_uploaded_csv(uploaded_file, max_rows: int) -> pd.DataFrame:
    """Parse an uploaded CSV into a frame, or explain precisely why it will not."""
    try:
        raw = uploaded_file.read()
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"The file could not be read: {exc}") from exc

    if not raw.strip():
        raise UploadError("The file is empty.")

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte string
        raise UploadError("The file is not text. Export it from your CRM as CSV.")

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            na_values=["", " ", "NA", "N/A", "na", "null", "NULL", "?", "-"],
        )
    except pd.errors.EmptyDataError as exc:
        raise UploadError("The file has no columns to read.") from exc
    except pd.errors.ParserError as exc:
        raise UploadError(
            "The file is not valid CSV. Check for stray quotes or a mismatched "
            f"number of columns. ({exc})"
        ) from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise UploadError("The file has a header row but no customers in it.")
    if len(frame) > max_rows:
        raise UploadError(
            f"The file holds {len(frame):,} rows; the limit is {max_rows:,}. "
            "Split it into smaller batches, or raise BDSS_MAX_UPLOAD_ROWS."
        )
    return frame


def guess_reference_column(frame: pd.DataFrame) -> str | None:
    """Find the column that names the customer, so results can be traced back."""
    candidates = [
        "customer_id", "CustomerId", "customerID", "customerId", "Customer_ID",
        "id", "ID", "customer_ref", "account_id", "record_id",
    ]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        if "id" in str(column).lower() and frame[column].is_unique:
            return str(column)
    return None


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def summarise(results: list[dict]) -> dict:
    """Aggregate a scored batch into the figures the campaign is planned from."""
    total = len(results)
    if not total:
        return {}

    bands = {"high": 0, "medium": 0, "low": 0}
    priorities = {"P1": 0, "P2": 0, "P3": 0}
    owners: dict[str, int] = {}
    interventions: dict[str, int] = {}
    categories: dict[str, float] = {category: 0.0 for category in CATEGORIES}
    flagged = 0

    for result in results:
        bands[result["band"]] = bands.get(result["band"], 0) + 1
        priorities[result["priority"]] = priorities.get(result["priority"], 0) + 1
        if result["predicted_churn"]:
            flagged += 1

        recommendation = result["recommendation"]
        owners[recommendation["owner"]] = owners.get(recommendation["owner"], 0) + 1
        interventions[recommendation["intervention"]] = (
            interventions.get(recommendation["intervention"], 0) + 1
        )
        for driver in result["drivers"]:
            categories[driver["category"]] = (
                categories.get(driver["category"], 0.0) + driver["abs_contribution"]
            )

    at_risk = bands["high"] + bands["medium"]
    return {
        "total": total,
        "flagged": flagged,
        "flagged_pct": round(100 * flagged / total, 1),
        "at_risk": at_risk,
        "at_risk_pct": round(100 * at_risk / total, 1),
        "bands": bands,
        "priorities": priorities,
        "owners": dict(sorted(owners.items(), key=lambda item: item[1], reverse=True)),
        "interventions": dict(
            sorted(interventions.items(), key=lambda item: item[1], reverse=True)[:8]
        ),
        "driver_categories": {
            category: round(share, 4)
            for category, share in sorted(
                categories.items(), key=lambda item: item[1], reverse=True
            )
            if share > 0
        },
        "mean_probability": round(
            sum(result["probability"] for result in results) / total, 4
        ),
        "caveated": sum(1 for result in results if result["recommendation"]["caveats"]),
    }


def sort_queue(results: list[dict]) -> list[dict]:
    """Order the work queue: P1 first, and within a priority, by score.

    This is the whole point of the decision layer. An unsorted list of at-risk
    customers is a report; a sorted one is a day's work.
    """
    order = {"P1": 0, "P2": 1, "P3": 2}
    return sorted(
        results,
        key=lambda result: (
            order.get(result["priority"], 9),
            -result["priority_score"],
            -result["probability"],
        ),
    )


def results_to_csv(results: list[dict], meta: dict) -> str:
    """Export the scored list for ingestion into the CRM.

    The export carries the model version and the drivers, not just the score.
    A scored list that leaves the building without its reasons is exactly the
    artefact this framework exists to stop producing.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "customer_ref", "churn_probability", "confidence", "predicted_label",
            "risk_band", "priority", "priority_score",
            "driver_1", "driver_1_effect", "driver_2", "driver_2_effect",
            "driver_3", "driver_3_effect",
            "recommended_intervention", "business_owner", "rationale", "caveats",
            "model", "model_version", "scored_at",
        ]
    )
    scored_at = meta.get("scored_at", "")
    for result in results:
        drivers = result["drivers"] + [{}, {}, {}]
        recommendation = result["recommendation"]
        writer.writerow(
            [
                result["customer_ref"],
                f"{result['probability']:.4f}",
                f"{result['confidence']:.4f}",
                result["predicted_label"],
                result["band_label"],
                result["priority"],
                result["priority_score"],
                drivers[0].get("label", ""), drivers[0].get("direction", ""),
                drivers[1].get("label", ""), drivers[1].get("direction", ""),
                drivers[2].get("label", ""), drivers[2].get("direction", ""),
                recommendation["intervention"],
                recommendation["owner"],
                recommendation["rationale"],
                " ".join(recommendation["caveats"]),
                meta.get("model_label", ""),
                meta.get("model_version", ""),
                scored_at,
            ]
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def _encode(figure) -> str:
    """Render a figure to a base64 PNG and release it."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def risk_distribution_chart(results: list[dict], threshold: float | None) -> str:
    """Where the book sits on the risk scale, and where the model draws the line."""
    probabilities = [result["probability"] for result in results]
    figure, axes = plt.subplots(figsize=(7.2, 3.4))

    counts, edges, patches = axes.hist(
        probabilities, bins=20, range=(0, 1), edgecolor=SURFACE, linewidth=1.2
    )
    # Colour by band, but the band boundaries are also legible on the axis, so
    # colour is never the only channel carrying the distinction.
    for count, patch, left in zip(counts, patches, edges[:-1]):
        centre = left + (edges[1] - edges[0]) / 2
        band = "high" if centre >= 0.7 else "medium" if centre >= 0.4 else "low"
        patch.set_facecolor(BAND_COLOURS[band])

    if threshold is not None:
        axes.axvline(threshold, color=INK_PRIMARY, linestyle="--", linewidth=1.5)
        axes.annotate(
            f"operating threshold {threshold:.2f}",
            xy=(threshold, axes.get_ylim()[1] * 0.92),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=9,
            fontweight="bold",
        )

    axes.set_xlabel("Calibrated probability of churn")
    axes.set_ylabel("Customers")
    axes.set_xlim(0, 1)
    axes.yaxis.grid(True, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_title(
        "Distribution of churn risk across the uploaded base",
        color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=12,
    )
    return _encode(figure)


def driver_category_chart(summary: dict) -> str:
    """Which kinds of driver are moving the risk across the whole batch.

    This is the campaign-level read: if contract terms dominate, the answer is a
    pricing review, not a hundred individual phone calls.
    """
    categories = summary.get("driver_categories") or {}
    if not categories:
        return ""

    total = sum(categories.values()) or 1.0
    labels = [CATEGORY_LABELS.get(name, name.title()) for name in categories]
    shares = [100 * value / total for value in categories.values()]
    colours = [CATEGORY_COLOURS.get(name, INK_MUTED) for name in categories]

    figure, axes = plt.subplots(figsize=(13.0, 0.42 * len(labels) + 1.6))
    positions = range(len(labels))
    bars = axes.barh(list(positions), shares, color=colours, height=0.62)
    axes.set_yticks(list(positions), labels)
    axes.invert_yaxis()
    axes.set_xlabel("Share of total attribution (%)")
    axes.set_xlim(0, max(shares) * 1.18)
    axes.xaxis.grid(True, linewidth=0.8)
    axes.set_axisbelow(True)

    # Every bar is directly labelled: two of these hues sit below 3:1 contrast on
    # a white surface, and a visible label is the required relief.
    for bar, share in zip(bars, shares):
        axes.annotate(
            f"{share:.0f}%",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
            fontweight="bold",
        )

    axes.set_title(
        "What is driving risk across this batch",
        color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=12,
    )
    return _encode(figure)


def priority_chart(summary: dict) -> str:
    """The shape of the work queue: how much is on fire, and how much can wait."""
    priorities = summary.get("priorities") or {}
    order = ["P1", "P2", "P3"]
    counts = [priorities.get(name, 0) for name in order]
    if not any(counts):
        return ""

    captions = ["P1 - act now", "P2 - act this cycle", "P3 - monitor"]
    figure, axes = plt.subplots(figsize=(7.2, 2.6))
    bars = axes.bar(
        captions, counts, color=[PRIORITY_COLOURS[name] for name in order], width=0.55
    )
    axes.set_ylabel("Customers")
    axes.set_ylim(0, max(counts) * 1.2 or 1)
    axes.yaxis.grid(True, linewidth=0.8)
    axes.set_axisbelow(True)

    for bar, count in zip(bars, counts):
        axes.annotate(
            f"{count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=INK_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )

    axes.set_title(
        "Recommended work queue by priority",
        color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=12,
    )
    return _encode(figure)


def global_importance_chart(importance: Iterable[dict], top_k: int = 12) -> str:
    """The model's leading drivers overall: the strategic view, not the case view."""
    drivers = list(importance)[:top_k]
    if not drivers:
        return ""

    labels = [driver["label"] for driver in drivers][::-1]
    values = [driver["mean_abs_shap"] for driver in drivers][::-1]
    colours = [CATEGORY_COLOURS.get(driver["category"], INK_MUTED) for driver in drivers][::-1]

    figure, axes = plt.subplots(figsize=(13.0, 0.34 * len(labels) + 1.5))
    positions = range(len(labels))
    axes.barh(list(positions), values, color=colours, height=0.62)
    axes.set_yticks(list(positions), labels)
    axes.set_xlabel("Mean |SHAP| (average effect on the churn estimate)")
    axes.xaxis.grid(True, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_title(
        "Leading churn drivers across the customer base",
        color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=12,
    )
    return _encode(figure)
