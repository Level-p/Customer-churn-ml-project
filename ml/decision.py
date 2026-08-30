"""Decision layer: from explanation to action.

This is the step that distinguishes the framework. Shapley drivers are grouped
into the four feature categories of the feature layer and matched against an
intervention taxonomy that names, for each driver, the response and the business
owner. Each rule carries a weight, and each recommendation carries a priority
derived from the model's probability band and the record's estimated value, so
that the output of the system is a ranked work queue rather than an
undifferentiated list.

The taxonomy lives in ``decision/intervention_rules.json`` -- an editable rule
table, not code -- so that planners can adjust it as offers and supply
constraints change, without a redeployment and without a developer. Every
recommendation records the drivers that produced it, which is what keeps the
decision trail auditable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from ml import features as feature_layer
from ml.config import RULES_PATH, get_logger

LOGGER = get_logger(__name__)

_WILDCARD = "*"


class RuleTableError(RuntimeError):
    """The rule table is missing or malformed."""


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime: float) -> dict:
    """Parse the rule table. Keyed on mtime so an edit is picked up live."""
    with Path(path).open("r", encoding="utf-8") as handle:
        table = json.load(handle)
    for key in ("rules", "priority_bands", "probability_bands"):
        if key not in table:
            raise RuleTableError(f"rule table is missing the '{key}' section")
    LOGGER.info("Loaded intervention taxonomy v%s (%d rules)",
                table.get("version", "?"), len(table["rules"]))
    return table


def load_rules(path: Path | str = RULES_PATH) -> dict:
    """Load the intervention taxonomy, re-reading it whenever the file changes."""
    path = Path(path)
    if not path.exists():
        raise RuleTableError(
            f"intervention rule table not found at {path}. It ships with the "
            "repository; restore it from version control before scoring."
        )
    return _load_cached(str(path), path.stat().st_mtime)


# --------------------------------------------------------------------------- #
# Banding
# --------------------------------------------------------------------------- #


def probability_band(probability: float, table: dict) -> dict:
    """Translate a probability into the banded score the dashboard shows.

    A planner acts on "high", not on 0.7431. The numeric probability is retained
    in the record for audit, but it is never the thing the interface leads with.
    """
    for band in table["probability_bands"]:
        if probability >= float(band["min_probability"]):
            return band
    return table["probability_bands"][-1]


def _value_multiplier(value: float | None, table: dict) -> float:
    """Scale priority by the record's estimated value, within stated bounds.

    Targeting the records whose treatment creates the most value is not the same
    as targeting those with the highest probability, which is why value enters
    the priority score and not just the risk estimate.
    """
    settings = table.get("value_scaling") or {}
    if value is None or not settings.get("enabled", False):
        return 1.0
    reference = float(settings.get("reference_value", 1.0)) or 1.0
    low = float(settings.get("min_multiplier", 0.5))
    high = float(settings.get("max_multiplier", 2.0))
    return max(low, min(high, float(value) / reference))


def priority_for(score: float, table: dict) -> dict:
    for band in table["priority_bands"]:
        if score >= float(band["min_score"]):
            return band
    return table["priority_bands"][-1]


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _applies_to(rule: dict, predicted_label: str) -> bool:
    targets = [str(target) for target in rule.get("applies_to", [_WILDCARD])]
    return _WILDCARD in targets or str(predicted_label) in targets


def _matches(rule: dict, driver: dict) -> bool:
    """Does this rule fire for this driver?

    A rule may key on a specific predictor (``feature``), on a whole feature
    category (``category``), or on both, and may additionally require that the
    driver push risk in a given direction.
    """
    match = rule.get("match", {})

    feature = match.get("feature")
    if feature:
        column = str(driver.get("column") or driver.get("feature", ""))
        if not (column == feature or column.startswith(f"{feature}_")):
            return False

    category = match.get("category")
    if category and driver.get("category") != category:
        return False

    direction = match.get("direction")
    if direction and driver.get("direction") != direction:
        return False

    if not feature and not category:
        return False
    return True


def recommend(
    predicted_label,
    probability: float,
    drivers: Sequence[dict],
    value: float | None = None,
    contested_features: Iterable[str] = (),
    table: dict | None = None,
) -> dict:
    """Turn one record's drivers into a prioritised, owned recommendation.

    Args:
        predicted_label: The class the model assigned to the record.
        probability: The calibrated probability of that class.
        drivers: Ranked SHAP drivers for the record, strongest first.
        value: Optional monetary value of the record, used to scale priority.
        contested_features: Features on which SHAP and LIME systematically
            disagreed during the audit. A recommendation resting on one of them
            is flagged for human attention rather than suppressed.
        table: Pre-loaded rule table; loaded from disk when omitted.

    Returns:
        The recommendation, the rule and driver that produced it, the priority,
        any secondary actions, and any caveats.
    """
    table = table or load_rules()
    contested = {str(name) for name in contested_features}
    rules = table["rules"]

    # A demographic driver may be shown -- suppressing it would make the
    # explanation a lie -- but it may never generate an offer. Rules are matched
    # against the actionable drivers only, so that a customer whose risk is
    # dominated by age or nationality still receives a recommendation grounded in
    # something the business can legitimately change, and the dominance itself is
    # raised as a flag rather than buried.
    actionable = [driver for driver in drivers if not _protected(driver)]
    protected = [driver for driver in drivers if _protected(driver)]

    matched: list[tuple[dict, dict]] = []
    for driver in actionable:
        for rule in rules:
            if _applies_to(rule, predicted_label) and _matches(rule, driver):
                matched.append((rule, driver))
                break  # strongest applicable rule per driver

    band = probability_band(float(probability), table)
    flags: list[str] = []

    if matched:
        rule, driver = matched[0]
    elif protected:
        rule = _rule_by_id(table, "demographic_no_offer") or table.get("default", {})
        driver = protected[0]
    else:
        rule = table.get("default", {})
        driver = drivers[0] if drivers else {}

    intervention = rule.get("intervention", "Manual review by a retention manager")
    owner = rule.get("owner", "Retention management")
    rationale = rule.get("rationale", "")
    weight = float(rule.get("weight", 0.5))
    rule_id = rule.get("id", "default")

    score = float(probability) * weight * _value_multiplier(value, table)
    priority = priority_for(score, table)

    caveats: list[str] = []
    # A caveat is raised only when the *recommendation itself* rests on a driver
    # the explanation audit disputed. Every disputed driver is still marked where
    # it is displayed -- the audit exists to say which attributions cannot be shown
    # without a health warning -- but a row-level warning that fires whenever any
    # of the three drivers is contested fires on most of the queue, and a warning
    # that appears on three customers in four is a warning nobody reads.
    if str(driver.get("feature")) in contested:
        caveats.append(
            f"This action rests on {driver.get('label')}, a driver on which SHAP and "
            "LIME disagreed during the explanation audit. Confirm before acting."
        )
    if protected and drivers and _protected(drivers[0]):
        flags.append("demographic_driver_dominant")
        caveats.append(
            f"The strongest driver of this estimate is {protected[0]['label']}, a "
            "demographic attribute. No offer is made on that basis; the recommendation "
            "rests on the strongest driver the business can legitimately act on."
            if matched else
            f"The strongest driver of this estimate is {protected[0]['label']}, a "
            "demographic attribute, and no actionable driver matched the taxonomy. "
            "Referred for human review."
        )
    if not matched and not protected:
        caveats.append("No taxonomy rule matched; routed for manual review.")

    return {
        "intervention": intervention,
        "owner": owner,
        "rationale": rationale,
        "rule_id": rule_id,
        "triggering_driver": driver.get("label"),
        "triggering_feature": driver.get("feature"),
        "priority": priority["name"],
        "priority_label": priority.get("label", priority["name"]),
        "priority_score": round(score, 4),
        "band": band["name"],
        "band_label": band.get("label", band["name"]),
        "band_colour": band.get("colour", "#6b7280"),
        "alternatives": [
            {
                "intervention": alternative["intervention"],
                "owner": alternative["owner"],
                "driver": alternative_driver.get("label"),
                "rule_id": alternative["id"],
            }
            for alternative, alternative_driver in matched[1:3]
        ],
        "caveats": caveats,
        "flags": flags,
        "taxonomy_version": table.get("version"),
    }


def _protected(driver: dict) -> bool:
    """Is this driver a demographic attribute?

    The flag is carried on the driver by the explanation layer, but it is
    recomputed from the feature name when absent, so that a rule table can never
    be tricked into making an offer on a protected attribute by a caller that
    forgot to set it.
    """
    if "protected" in driver:
        return bool(driver["protected"])
    return feature_layer.is_protected(str(driver.get("feature", "")))


def _rule_by_id(table: dict, rule_id: str) -> dict | None:
    return next((rule for rule in table["rules"] if rule["id"] == rule_id), None)


def taxonomy_table(table: dict | None = None) -> list[dict]:
    """The rule table rendered for display: the taxonomy is published, not hidden.

    Publishing it is a fairness safeguard as much as a usability one. If offers
    flow only to the records a model flags, the groups it never flags quietly
    receive worse treatment; an open taxonomy makes that visible.
    """
    table = table or load_rules()
    rows = []
    for rule in table["rules"]:
        match = rule.get("match", {})
        trigger = match.get("feature") or match.get("category", "")
        direction = match.get("direction")
        rows.append(
            {
                "id": rule["id"],
                "driver": trigger.replace("_", " ").title(),
                "category": match.get("category", ""),
                "direction": direction or "any",
                "applies_to": ", ".join(str(t) for t in rule.get("applies_to", [_WILDCARD])),
                "intervention": rule["intervention"],
                "owner": rule["owner"],
                "weight": rule.get("weight", 1.0),
                "rationale": rule.get("rationale", ""),
            }
        )
    return rows
