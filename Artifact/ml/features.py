"""Feature layer: categories, derived predictors and business-readable names.

Predictors are organised into five categories that serve both as an engineering
checklist and as the vocabulary of the explanation layer. The dashboard never
shows a raw column name to a retention manager; it shows the friendly label and
the category resolved here, because "Contract: month-to-month" is something a
retention team can act on and ``cat__subscription_type_Basic`` is not.

The categories follow the research design:

    behavioural    login and session frequency, product usage, breadth of adoption
    transactional  balances, fees, payment method, salary, credit standing
    contractual    tenure, product holdings, subscription tier, value-added services
    engagement     complaints, support contact, satisfaction proxies
    demographic    age, gender, geography, device -- audited, never acted on

The demographic category is kept separate for a reason that is ethical rather
than technical. Those attributes are permitted as predictors only so that the
fairness audit can test whether they are functioning as proxies; the decision
layer is forbidden from turning one into a differential offer.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

BEHAVIOURAL = "behavioural"
TRANSACTIONAL = "transactional"
CONTRACTUAL = "contractual"
ENGAGEMENT = "engagement"
DEMOGRAPHIC = "demographic"
UNCATEGORISED = "uncategorised"

CATEGORIES: tuple[str, ...] = (
    BEHAVIOURAL,
    TRANSACTIONAL,
    CONTRACTUAL,
    ENGAGEMENT,
    DEMOGRAPHIC,
    UNCATEGORISED,
)

CATEGORY_LABELS: dict[str, str] = {
    BEHAVIOURAL: "Usage and activity",
    TRANSACTIONAL: "Money and payments",
    CONTRACTUAL: "Contract and products",
    ENGAGEMENT: "Service and engagement",
    DEMOGRAPHIC: "Demographic",
    UNCATEGORISED: "Other",
}

#: Categories on which no automated differential offer may be made.
PROTECTED_CATEGORIES: frozenset[str] = frozenset({DEMOGRAPHIC})

#: Ordered rules mapping a raw column to a category, matched case-insensitively.
#: The first match wins, so specific patterns precede general ones.
CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^(isactivemember|is_active|active_member)", BEHAVIOURAL),
    (r"^(watch_hours|avg_watch|watch_|last_login|number_of_profiles|profiles_)", BEHAVIOURAL),
    (r"^(sessions?|logins?|page_views|browsing|usage|activity|visits?|"
     r"days_since|recency|total_orders|orders_per|streams?_)", BEHAVIOURAL),
    (r"^(balance|estimatedsalary|salary|creditscore|credit_score|monthly_fee|fee_|"
     r"charge|payment_method|paymentmethod|avg_order_value|spend|revenue|"
     r"discount|cashback|price_sensitivity|income)", TRANSACTIONAL),
    (r"^(tenure|numofproducts|num_products|products_|subscription_type|contract|"
     r"hascrcard|has_cr_card|plan_|tier|renewal|value_added|onlinesecurity|"
     r"techsupport|account_age)", CONTRACTUAL),
    (r"^(complaint|support|ticket|satisfaction|nps|review|churn_risk_flag|"
     r"return_rate|cart_abandonment|engagement_score|campaign|email_open)", ENGAGEMENT),
    (r"^(age|gender|geography|region|country|device|favorite_genre|marital|"
     r"seniorcitizen|dependents|partner|location)", DEMOGRAPHIC),
)

#: Business-readable labels. Anything absent falls back to a title-cased name, so
#: an unfamiliar CSV degrades gracefully rather than failing.
FRIENDLY_NAMES: dict[str, str] = {
    # Retail banking
    "CreditScore": "Credit score",
    "Geography": "Country",
    "Gender": "Gender",
    "Age": "Age",
    "Tenure": "Years as a customer",
    "Balance": "Account balance",
    "NumOfProducts": "Products held",
    "HasCrCard": "Holds a credit card",
    "IsActiveMember": "Active member",
    "EstimatedSalary": "Estimated salary",
    # Subscription streaming
    "age": "Age",
    "gender": "Gender",
    "region": "Region",
    "device": "Primary device",
    "subscription_type": "Subscription tier",
    "watch_hours": "Hours watched",
    "last_login_days": "Days since last login",
    "monthly_fee": "Monthly fee",
    "payment_method": "Payment method",
    "number_of_profiles": "Profiles on the account",
    "avg_watch_time_per_day": "Average daily viewing",
    "favorite_genre": "Preferred genre",
    # Derived
    "balance_per_product": "Balance per product held",
    "balance_to_salary": "Balance against salary",
    "zero_balance": "Empty account",
    "products_per_tenure": "Products taken up per year",
    "tenure_ratio": "Share of adult life as a customer",
    "watch_per_fee": "Viewing per pound paid",
    "watch_per_profile": "Viewing per profile",
    "fee_per_profile": "Fee per profile",
    "engagement_decay": "Viewing against time since login",
    "is_dormant": "Dormant (no login for 30 days)",
}

_ONE_HOT_SPLIT = re.compile(r"^(?P<column>.+?)_(?P<level>[^_]+)$")


def categorise(feature: str) -> str:
    """Return the category of a raw or encoded feature name."""
    raw = base_column(feature).lower()
    for pattern, category in CATEGORY_PATTERNS:
        if re.match(pattern, raw):
            return category
    return UNCATEGORISED


def is_protected(feature: str) -> bool:
    """Is this predictor a demographic attribute, or a one-hot level of one?"""
    return categorise(feature) in PROTECTED_CATEGORIES


def base_column(feature: str) -> str:
    """Strip pipeline and one-hot decoration from an encoded feature name.

    ``num__Balance``                 -> ``Balance``
    ``cat__subscription_type_Basic`` -> ``subscription_type``
    """
    name = feature
    if "__" in name:
        name = name.split("__", 1)[1]
    return name


def friendly_name(feature: str, known_columns: Iterable[str] | None = None) -> str:
    """Return a business-readable label for a raw or encoded feature name.

    One-hot columns render as ``Label: level``, so a retention manager reads
    "Subscription tier: Basic" rather than ``cat__subscription_type_Basic``.
    """
    name = base_column(feature)
    if name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[name]

    columns = {base_column(column) for column in (known_columns or ())}
    columns |= set(FRIENDLY_NAMES)
    # Longest matching prefix wins: payment_method_Credit_Card -> payment_method.
    candidates = [column for column in columns if name.startswith(f"{column}_")]
    if candidates:
        column = max(candidates, key=len)
        level = name[len(column) + 1:]
        label = FRIENDLY_NAMES.get(column, _titleise(column))
        return f"{label}: {_titleise(level)}"

    match = _ONE_HOT_SPLIT.match(name)
    if match and match.group("column") in FRIENDLY_NAMES:
        return (
            f"{FRIENDLY_NAMES[match.group('column')]}: "
            f"{_titleise(match.group('level'))}"
        )
    return _titleise(name)


def _titleise(name: str) -> str:
    cleaned = name.replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def describe(feature: str, known_columns: Iterable[str] | None = None) -> dict[str, object]:
    """The full presentation record for a feature, as the dashboard needs it."""
    category = categorise(feature)
    return {
        "feature": feature,
        "column": base_column(feature),
        "label": friendly_name(feature, known_columns),
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "protected": category in PROTECTED_CATEGORIES,
    }


# --------------------------------------------------------------------------- #
# Derived predictors
# --------------------------------------------------------------------------- #


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the ratio and recency predictors the evidence favours.

    Dynamic and relative quantities carry signal that raw levels do not: a
    balance means one thing to a customer earning 30,000 and another to one
    earning 300,000, and viewing hours mean one thing on a 6.99 plan and another
    on a 19.99 plan. Every derivation is guarded on the presence of its inputs,
    so the function is safe to call on an arbitrary uploaded CSV that carries only
    a subset of the columns -- which is exactly what the dashboard does.
    """
    df = frame.copy()
    eps = 1e-6

    def has(*columns: str) -> bool:
        return all(column in df.columns for column in columns)

    # --- Retail banking ---
    if has("Balance", "NumOfProducts"):
        df["balance_per_product"] = df["Balance"] / (df["NumOfProducts"] + eps)
    if has("Balance", "EstimatedSalary"):
        df["balance_to_salary"] = df["Balance"] / (df["EstimatedSalary"] + eps)
    if has("Balance"):
        # A zero balance is not a small balance: it is a different state, and a
        # linear model cannot express that without being told.
        df["zero_balance"] = (df["Balance"] <= 0).astype(int)
    if has("NumOfProducts", "Tenure"):
        df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)
    if has("Tenure", "Age"):
        df["tenure_ratio"] = df["Tenure"] / (df["Age"] - 17.0).clip(lower=1.0)

    # --- Subscription streaming ---
    if has("watch_hours", "monthly_fee"):
        df["watch_per_fee"] = df["watch_hours"] / (df["monthly_fee"] + eps)
    if has("watch_hours", "number_of_profiles"):
        df["watch_per_profile"] = df["watch_hours"] / (df["number_of_profiles"] + eps)
    if has("monthly_fee", "number_of_profiles"):
        df["fee_per_profile"] = df["monthly_fee"] / (df["number_of_profiles"] + eps)
    if has("avg_watch_time_per_day", "last_login_days"):
        df["engagement_decay"] = df["avg_watch_time_per_day"] / (df["last_login_days"] + 1.0)
    if has("last_login_days"):
        df["is_dormant"] = (df["last_login_days"] >= 30).astype(int)

    derived = [
        column
        for column in (
            "balance_per_product", "balance_to_salary", "zero_balance",
            "products_per_tenure", "tenure_ratio", "watch_per_fee",
            "watch_per_profile", "fee_per_profile", "engagement_decay", "is_dormant",
        )
        if column in df.columns
    ]
    if derived:
        # A pathological denominator must not cost a customer their score.
        df[derived] = (
            df[derived]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-1e9, upper=1e9)
        )
    return df


def category_summary(features: Iterable[str]) -> dict[str, int]:
    """Count features per category, for the training log and the dashboard."""
    counts = {category: 0 for category in CATEGORIES}
    for feature in features:
        counts[categorise(feature)] += 1
    return {category: count for category, count in counts.items() if count}
