# Retention BDSS — Lab Manual

### A Business Decision-Support System for Interpretable Churn Prediction

**Codebase:** `Artifact/`
**Stack:** Python 3.13 · Django 4.2 LTS · scikit-learn · XGBoost · SHAP · LIME
**Companion paper:** *Predicting Customer Churn in Subscription Services Using Interpretable Machine Learning: A Business Decision-Support Framework for Retention Management*

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [The problem this system solves](#11-the-problem-this-system-solves)
   - 1.2 [What we actually built](#12-what-we-actually-built)
   - 1.3 [How to read this manual](#13-how-to-read-this-manual)
2. [Getting started](#2-getting-started)
   - 2.1 [Installation](#21-installation)
   - 2.2 [Training the models](#22-training-the-models)
   - 2.3 [Running the web application](#23-running-the-web-application)
   - 2.4 [Running the tests](#24-running-the-tests)
3. [Architecture](#3-architecture)
   - 3.1 [The five layers](#31-the-five-layers)
   - 3.2 [Repository map](#32-repository-map)
   - 3.3 [The single most important structural decision](#33-the-single-most-important-structural-decision)
4. [The data layer](#4-the-data-layer)
   - 4.1 [`ml/config.py` — everything that must be reproducible](#41-mlconfigpy--everything-that-must-be-reproducible)
   - 4.2 [The datasets, and the one we refused to use](#42-the-datasets-and-the-one-we-refused-to-use)
   - 4.3 [`ml/data_prep.py` — cleaning, encoding, splitting](#43-mldata_preppy--cleaning-encoding-splitting)
   - 4.4 [Why SMOTE lives inside the pipeline](#44-why-smote-lives-inside-the-pipeline)
5. [The feature layer](#5-the-feature-layer)
   - 5.1 [Five categories, and why the fifth is special](#51-five-categories-and-why-the-fifth-is-special)
   - 5.2 [Derived predictors](#52-derived-predictors)
   - 5.3 [Speaking the business's language](#53-speaking-the-businesss-language)
6. [The model layer](#6-the-model-layer)
   - 6.1 [The shared seven-stage protocol](#61-the-shared-seven-stage-protocol)
   - 6.2 [Logistic regression — the baseline that never leaves](#62-logistic-regression--the-baseline-that-never-leaves)
   - 6.3 [Random forest](#63-random-forest)
   - 6.4 [XGBoost](#64-xgboost)
   - 6.5 [Calibration](#65-calibration)
   - 6.6 [The operating threshold is a business decision](#66-the-operating-threshold-is-a-business-decision)
   - 6.7 [Evaluation metrics](#67-evaluation-metrics)
   - 6.8 [Choosing between the three](#68-choosing-between-the-three)
7. [The explanation layer](#7-the-explanation-layer)
   - 7.1 [SHAP, and the shapes nobody warns you about](#71-shap-and-the-shapes-nobody-warns-you-about)
   - 7.2 [Auditing SHAP against LIME](#72-auditing-shap-against-lime)
   - 7.3 [What the audit actually found](#73-what-the-audit-actually-found)
8. [The fairness audit](#8-the-fairness-audit)
9. [The decision layer](#9-the-decision-layer)
   - 9.1 [The intervention taxonomy](#91-the-intervention-taxonomy)
   - 9.2 [The rule engine](#92-the-rule-engine)
   - 9.3 [The demographic invariant](#93-the-demographic-invariant)
   - 9.4 [Priority: risk weighted by value](#94-priority-risk-weighted-by-value)
10. [The model registry](#10-the-model-registry)
11. [The web application](#11-the-web-application)
    - 11.1 [Settings and security](#111-settings-and-security)
    - 11.2 [Authentication](#112-authentication)
    - 11.3 [`ml_engine/predictor.py` — the inference path](#113-ml_enginepredictorpy--the-inference-path)
    - 11.4 [`ml_engine/utils.py` — intake, summaries, charts](#114-ml_engineutilspy--intake-summaries-charts)
    - 11.5 [The audit trail (`dashboard/models.py`)](#115-the-audit-trail-dashboardmodelspy)
    - 11.6 [Overview](#116-overview)
    - 11.7 [Upload](#117-upload)
    - 11.8 [Results — the retention queue](#118-results--the-retention-queue)
    - 11.9 [Model cards](#119-model-cards)
    - 11.10 [Taxonomy and registry](#1110-taxonomy-and-registry)
    - 11.11 [The REST API](#1111-the-rest-api)
    - 11.12 [The admin log](#1112-the-admin-log)
12. [Testing](#12-testing)
13. [Results, and what they mean](#13-results-and-what-they-mean)
14. [Limitations and future work](#14-limitations-and-future-work)
15. [Glossary](#15-glossary)
16. [Conclusion and recommendations](#16-conclusion-and-recommendations)
- [Appendix A: File map](#appendix-a-file-map)
- [Appendix B: Configuration reference](#appendix-b-configuration-reference)

---

## 1. Introduction

### 1.1 The problem this system solves

Imagine you are a retention manager at a bank. On Monday morning a data scientist emails you a spreadsheet. It has ten thousand rows, and each row has a customer number and a number like `0.82`. That is all.

What do you do with it?

You cannot phone ten thousand people. You do not know whether `0.82` means this customer is angry about a fee, bored of the product, or simply old — and each of those needs a completely different response. You do not know whether to send a discount, a phone call, or an apology. And when your director asks *why* customer 15634602 was offered three months free, you have nothing to say except "the computer said so."

That is the gap this system is built to close. The churn-prediction literature has largely solved the accuracy problem — gradient boosting wins on tabular data, and has done for years. What it has not solved is the last mile: turning a probability into a decision that a human being can execute, justify, and be held accountable for. Explanations, where they exist at all, tend to be produced *for the data scientist* and stop there.

So the system in this repository does not stop at `0.82`. For every customer it returns:

- a **calibrated probability**, banded into high / medium / low so nobody has to interpret a decimal;
- the **three strongest drivers** of that estimate, in business language ("Years as a customer", not `num__Tenure`);
- a **recommended action** with a named business owner ("Structured onboarding contact within the first 90 days" → Customer success);
- a **priority**, so the output is a ranked work queue rather than an undifferentiated list;
- the **model version** that produced all of it, so the decision can be reconstructed months later.

And — this is the part most systems skip — it tells you when it doesn't trust itself.

### 1.2 What we actually built

Two halves, in one repository:

**The pipeline** (`ml/`) trains three classifiers of increasing complexity on two real churn datasets, explains every prediction with Shapley values, audits those explanations against a second, independent explainer, audits the model for demographic bias, and registers the result with everything needed to reproduce it.

**The web application** (`bdss_project/`) is a Django service where a retention analyst signs in, uploads a customer list, picks a model, and gets a ranked queue back — with a REST API alongside it for the CRM to call, and an audit log underneath that records every score served and every human override.

Roughly 6,450 lines of Python and 1,000 lines of templates. Fourteen tests.

### 1.3 How to read this manual

Sections 4–10 follow the data as it flows through the pipeline: raw CSV → cleaned features → trained model → explanation → recommendation → registry. Section 11 then walks the web application, using screenshots of the running system. If you only have twenty minutes, read §1, §3, §9.3 (the demographic invariant), §7.3 (what the explanation audit found) and §13 (results) — those carry the ideas that make this system different from a Kaggle notebook with a web page bolted on.

Every file path in this manual is real and every number is measured, not illustrative.

---

## 2. Getting started

### 2.1 Installation

```bash
cd Artifact
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned to exact versions. This is not pedantry. A trained model here is a pickled object graph containing scikit-learn, imbalanced-learn and XGBoost internals; unpickling it under a different minor version is a *silent* failure risk, not a loud one. The version that trained the artifact is recorded inside the artifact for exactly this reason.

### 2.2 Training the models

```bash
python -m ml.train_all --dataset bank_churn --max-rows 20000
python -m ml.train_all --dataset netflix_churn
```

`train_all` trains all three classifiers, prints the comparison table, and recommends one. Individual models: `python -m ml.train_logistic_regression`, `-m ml.train_random_forest`, `-m ml.train_xgboost`.

A word on runtime, because it will surprise you. On a 20,000-row banking sample the three models took **30 seconds, 22 minutes, and 3 minutes** respectively. The random forest is the slow one by an order of magnitude — its grid search is 16 candidates × 10 folds = 160 forest fits, each on ~26,000 SMOTE-balanced rows. That is the honest cost of an exhaustive grid search, not a bug. Two escape hatches exist: `--fast` (reduced search space, whole ladder in about two minutes) and `--max-rows N` (stratified subsample).

### 2.3 Running the web application

```bash
cd bdss_project
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open **http://127.0.0.1:8000/**. The development account created during the build is `admin` / `retention2026` — change it before this goes near real data.

### 2.4 Running the tests

```bash
cd bdss_project
python manage.py test dashboard
```

---

## 3. Architecture

### 3.1 The five layers

The system is a pipeline of five layers, each consuming the output of the one above it. This mirrors the architecture in the companion paper, and the code is organised so you can point at a module for every box in the diagram.

```
  DATA LAYER          ml/data_prep.py       clean, encode, scale, drop redundant predictors,
                      ml/config.py          80/20 stratified split, SMOTE inside the fold
        |
  FEATURE LAYER       ml/features.py        behavioural / transactional / contractual /
                                            engagement / demographic + derived ratios
        |
  MODEL LAYER         ml/train_common.py    10-fold CV, tuning, isotonic calibration,
                      ml/train_*.py         threshold chosen on validation folds
        |
  EXPLANATION LAYER   ml/explain.py         SHAP per customer (TreeExplainer),
                      ml/fairness.py        audited against LIME; bias audited by group
        |
  DECISION LAYER      ml/decision.py        intervention taxonomy -> action, owner, priority
                      bdss_project/         dashboard, batch scoring, REST API, audit trail
```

### 3.2 Repository map

```
Artifact/
├── ml/                              the pipeline
│   ├── config.py                    paths, seeds, protocol constants, datasets, economics
│   ├── data_prep.py                 the data layer
│   ├── features.py                  categories, derived predictors, business-readable names
│   ├── calibration.py               isotonic calibration on out-of-fold probabilities
│   ├── evaluation.py                precision, recall, F1, ROC-AUC, MCC, Brier, threshold, profit
│   ├── explain.py                   SHAP, and the LIME audit of SHAP
│   ├── fairness.py                  per-group error rates, four-fifths check, proxy detection
│   ├── decision.py                  the rule engine over the intervention taxonomy
│   ├── registry.py                  versioned model registry
│   ├── train_common.py              the shared seven-stage training protocol
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   └── train_all.py                 trains the ladder, prints the dual-axis comparison
│
├── decision/intervention_rules.json the editable taxonomy — a data file, not code
├── datasets/                        source data (Bank, netflix, Eshop)
├── models/                          trained artifacts + registry.json
├── reports/                         evaluation reports and model comparisons
│
└── bdss_project/                    the Django service
    ├── dashboard/                   models, views, forms, API, templates, tests
    └── ml_engine/                   predictor.py (inference), utils.py (intake + charts)
```

### 3.3 The single most important structural decision

**The web tier imports `ml/`. It does not reimplement it.**

`bdss_project/bdss_project/settings.py` puts the repository root on the import path:

```python
# bdss_project/bdss_project/settings.py
REPO_ROOT = Path(os.environ.get("BDSS_REPO_ROOT", BASE_DIR.parent))

# The `ml` package is the single source of truth for the feature vocabulary,
# the calibration maths and the intervention taxonomy. Importing it keeps the
# served explanation identical to the trained one; re-implementing it here would
# guarantee the two drift apart.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

The alternative — and it is the common one — is a web app that re-derives features, re-applies scaling, and re-maps drivers to labels in its own code. That works for about six weeks. Then someone adds a derived feature to the training script, forgets the serving code, and the model is quietly scored on a feature vector that means something subtly different from the one it was trained on. Nothing throws. The metrics on the model card stay beautiful. The predictions are wrong.

By importing the same modules, the feature vocabulary, the calibration function and the intervention taxonomy have exactly **one** definition each. What the dashboard shows a retention manager is, by construction, what the training run measured.

The cost is honest: the web tier now depends on scikit-learn, XGBoost, imbalanced-learn and SHAP at runtime (about 700 MB of virtual environment), and you cannot deploy the dashboard without the pipeline code beside it. For a decision-support system where correctness of the explanation is the entire product, that is a trade worth making.

---

## 4. The data layer

### 4.1 `ml/config.py` — everything that must be reproducible

Every path, seed and protocol constant lives in one file, so a run can be reproduced from the values recorded in a model artifact. The interesting part is the economics:

```python
# ml/config.py
@dataclass(frozen=True)
class Economics:
    value_true_positive: float = 100.0   # retained customer value, net of offer
    cost_false_positive: float = 15.0    # retention offer wasted on a loyal customer
    cost_false_negative: float = 200.0   # customer lost, and their revenue with them
    value_true_negative: float = 0.0     # business as usual
```

A churn model is worth what the retention campaign it induces is worth, so the confusion matrix is **valued**, not merely counted. These four numbers are placeholders, and the manual says so loudly because the code does: they are the first thing an adopting firm must replace, since they — not the model — decide where the operating threshold lands. Set the cost of losing a customer high enough and the system will contact everyone; set it low and it will contact nobody. That is a commercial decision wearing a technical disguise, and §6.6 is about dragging it into the open.

### 4.2 The datasets, and the one we refused to use

| Dataset | Rows | Label | Positive rate | Used? |
|---|---|---|---|---|
| `datasets/Bank/train.csv` | 165,034 | `Exited` | 21.2% | **Yes** — retail banking |
| `datasets/netflix/netflix_customer_churn.csv` | 5,000 | `churned` | 50.3% | **Yes** — subscription streaming |
| `datasets/Eshop/ecommerce_customer_features.csv` | 6,000 | *none* | — | **No** |
| `datasets/Bank/test.csv` | 110,023 | *none* | — | Unlabelled; used to demo batch scoring |

The e-commerce file has **no target column** — fifteen feature columns and nothing to predict. The obvious fix is to define churn as "no purchase in N days", and we very nearly did. Then we measured it:

- `engagement_score` correlates with `days_since_last_purchase` at **r = −0.85**. Keep both, and the model reads the answer off the back of the exam paper: a ~0.99 AUC that is pure leakage.
- Remove both, and the remaining twelve predictors correlate with the label at **|r| ≤ 0.024**. A coin flip.

There is no honest model in that file. Fabricating a third domain would have looked better in a results table and been worth nothing, so the dataset is deliberately **not registered** in `DATASETS`, and the reason is written into the config where the next engineer will find it:

```python
# ml/config.py
    # The e-commerce file in datasets/Eshop carries no label column, so it
    # supports no supervised model and is deliberately not registered here. See
    # the limitations section of the README: a churn label defined from
    # `days_since_last_purchase` would either leak through `engagement_score`
    # (r = -0.85 with the cutoff column) or, with both removed, leave predictors
    # that are uncorrelated with the label. Fabricating a third domain is worse
    # than reporting two.
```

One more decision worth flagging. In the banking data, `Surname` is **dropped**, not merely unused:

```python
        # 'Surname' is dropped rather than merely unused. It is a direct
        # identifier and a proxy for ethnicity and nationality, and no model in
        # a retention system has any business reading one.
        id_columns=("id", "CustomerId"),
        drop_columns=("Surname",),
```

A gradient-boosted tree will happily learn that customers named *Okwudilichukwu* churn at a different rate from customers named *Smith*. It will improve your F1. It is also indefensible, and it will not be in the model, so it cannot be.

### 4.3 `ml/data_prep.py` — cleaning, encoding, splitting

The same protocol runs on every dataset so that cross-domain comparisons are not confounded by different preprocessing: duplicates removed, continuous fields median-imputed, categorical fields given an explicit `Missing` level, one-hot encoding, min-max scaling to the unit interval, an 80/20 stratified split.

Two details are worth pausing on.

**Redundant predictors are removed on the training partition only.** Near-zero-variance columns go, and of any pair correlated above 0.9 one is dropped — because strongly correlated predictors destabilise logistic regression coefficients *and* dilute Shapley attributions across duplicates (see §7.3, where exactly this shows up in the audit). Crucially, the selection is computed on training data and then applied verbatim to test data and to anything scored later, so the decision is part of the model, not part of the data.

**pandas 3 broke a common idiom, quietly.** The usual way to find text columns is `df[col].dtype == object`. Under pandas 3 a text column has a `str` dtype, not `object`, so that test silently returns `False` and every categorical column is treated as numeric. The fix is to ask the question that was actually meant:

```python
# ml/data_prep.py
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
```

### 4.4 Why SMOTE lives inside the pipeline

This is the single most commonly botched step in the entire churn literature, so it gets its own subsection.

Churners are a minority — 21% in the banking data. SMOTE fixes this by inventing synthetic minority examples by interpolating between real ones. The trap is *when* you do it. If you oversample the whole training set and then cross-validate, a synthetic customer generated from real customer #47 can land in the training fold while customer #47 sits in the validation fold. The model has effectively seen the answer. Your cross-validated scores go up, your production scores do not, and nothing anywhere throws an error.

The defence here is structural rather than disciplinary — SMOTE is a *step in the pipeline object*:

```python
# ml/data_prep.py
def build_pipeline(estimator, numeric, categorical, y_train=None, seed=RANDOM_SEED, resample=True):
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
```

Because it is an `imblearn.Pipeline`, the sampler fires on `fit` and is bypassed on `predict`. Hand this object to `GridSearchCV` and the oversampling happens *inside each fold*, automatically, forever, whether or not the next engineer remembers that it should. The test partition keeps its natural class distribution throughout, because performance claims are only credible against the imbalance the model will actually face.

`make_sampler` also degrades gracefully: if the rarest class has fewer than two members SMOTE is disabled with a warning rather than raising, because a pipeline that explodes on a pathological slice is worse than one that says so and carries on.

---

## 5. The feature layer

### 5.1 Five categories, and why the fifth is special

Every predictor is sorted into one of five categories (`ml/features.py`). Four of them are an engineering checklist and a vocabulary for the explanation layer:

| Category | What it holds | Shown to users as |
|---|---|---|
| `behavioural` | activity, usage, logins, viewing | "Usage and activity" |
| `transactional` | balances, fees, salary, credit standing | "Money and payments" |
| `contractual` | tenure, products held, subscription tier | "Contract and products" |
| `engagement` | complaints, support contact, satisfaction | "Service and engagement" |
| `demographic` | age, gender, geography, device | "Demographic" |

The fifth exists for an ethical reason, not a technical one:

```python
# ml/features.py
#: Categories on which no automated differential offer may be made.
PROTECTED_CATEGORIES: frozenset[str] = frozenset({DEMOGRAPHIC})
```

Demographic attributes are permitted as *predictors* — partly because they genuinely carry signal, and partly so the fairness audit can test whether they are functioning as proxies. But the decision layer is forbidden from turning one into an offer. §9.3 shows how that is enforced.

### 5.2 Derived predictors

Ratios and recency variables carry signal that raw levels do not. A balance of £120,000 means one thing to a customer earning £30,000 and something quite different to one earning £300,000; forty hours of viewing means one thing on a £6.99 plan and another on a £19.99 plan. So `engineer_features` adds `balance_to_salary`, `products_per_tenure`, `watch_per_fee`, `engagement_decay`, and a handful of others.

Two guarded details:

```python
# ml/features.py
    if has("Balance"):
        # A zero balance is not a small balance: it is a different state, and a
        # linear model cannot express that without being told.
        df["zero_balance"] = (df["Balance"] <= 0).astype(int)
```

Every derivation is guarded on the presence of its inputs, which is what allows the *dashboard* to call this same function on an arbitrary uploaded CSV that carries only a subset of the columns. And every derived column is clipped, because a pathological denominator must not cost a customer their score:

```python
        df[derived] = (
            df[derived]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-1e9, upper=1e9)
        )
```

### 5.3 Speaking the business's language

`friendly_name()` turns `cat__Geography_Germany` into "Country: Germany" and `num__Tenure` into "Years as a customer". It handles the pipeline's `num__`/`cat__` prefixes, splits one-hot columns back into their column and level, and falls back to a title-cased name for anything it has never seen — so an unfamiliar CSV degrades gracefully instead of crashing.

This function is small and unglamorous, and it is the reason the results page is readable by someone who has never heard of one-hot encoding.

---

## 6. The model layer

### 6.1 The shared seven-stage protocol

All three models run through **one** driver (`ml/train_common.py`), so the comparison between them measures the models and not the diligence with which each was tuned. The stages are logged as they run, which makes the console output a readable narrative rather than a wall of numbers:

1. **Data layer** — prepare, split, categorise features.
2. **Model layer** — hyperparameter search, 10-fold CV, SMOTE in-fold.
3. **Calibration** — isotonic, on out-of-fold probabilities.
4. **Decision layer** — pick the operating threshold on those same validation folds.
5. **Refit** — on the full training partition.
6. **Evaluation** — a single pass over the untouched test partition.
7. **Explanation** — SHAP, audited against LIME; then the fairness audit.

One subtlety inside `train()` that took a while to get right. XGBoost refuses non-integer class labels, so every model is fitted on integer-encoded labels — and once you do that, *the mapping from probability column to class label becomes load-bearing*. Get it backwards and every recommendation in the system is inverted, with no error anywhere. So the invariant is asserted, not assumed:

```python
# ml/train_common.py
    # The classifier is fitted on integer-encoded labels: XGBoost requires them,
    # and using them everywhere keeps one probability-column convention across
    # the three models. Column i of every probability matrix in this pipeline is
    # the class ``encoder.classes_[i]``, and nothing downstream may assume more.
    encoder = LabelEncoder().fit(pd.concat([data.y_train, data.y_test]))
    ...
    if not np.array_equal(np.asarray(fitted.classes_), np.arange(len(classes))):
        raise RuntimeError(
            "the fitted classifier's class order does not match the label encoding; "
            "probability columns cannot be trusted"
        )
```

The same care goes into the tuning objective. scikit-learn's `average_precision` scorer assumes the positive class is `1`. If your positive label happens to sort first — "Churn" before "Retained" — it will cheerfully optimise the wrong class and hand you a beautifully tuned model that predicts loyalty. So the scorer pins `pos_label` explicitly:

```python
# ml/train_common.py
def scoring_for(task: str, focus_index: int = 1):
    """...
    ``pos_label`` is pinned to the encoded index of the positive class rather
    than left at scikit-learn's default of 1. Where the positive label happens to
    sort first -- "Churn" before "Retained", say -- the default would quietly
    optimise the wrong class.
    """
    if task != "binary":
        return "f1_macro"
    return make_scorer(average_precision_score, response_method="predict_proba",
                       pos_label=focus_index)
```

Average precision, not ROC-AUC, is the search objective: it is threshold-free (so the search cannot bake in an operating point that is chosen later, by the business) and it does not flatter a model whose positive class is rare.

### 6.2 Logistic regression — the baseline that never leaves

`ml/train_logistic_regression.py`. An L2-penalised linear model, tuned over `C` and `class_weight` by grid search.

It is not here to win. It is here as the reference against which the cost of opacity is priced: **if the ensembles cannot beat it by a margin that matters to the business, the business should be running this.** Every coefficient is a log-odds contribution that can be read directly, with no post-hoc machinery at all — which is why the script prints them:

```python
# ml/train_logistic_regression.py
def log_coefficients(outcome: TrainingOutcome, top_k: int = 15) -> None:
    """Print the log-odds contributions: the model explaining itself.

    This is the property the ensembles have to buy back with SHAP, and printing
    it beside the Shapley ranking is the cheapest sanity check in the pipeline --
    a linear coefficient and a Shapley value that disagree about the sign of a
    driver is a signal worth chasing before anyone trusts the score.
    """
```

That last sentence is the whole point of keeping it. On the banking data it reports `Age` at +4.90 log-odds (odds ×134) and `Products held` at −2.12 — and SHAP, on the ensembles, ranks exactly those two first and second, in the same directions. Two completely different methods agreeing is worth more than either one alone.

### 6.3 Random forest

`ml/train_random_forest.py`. Grid search over the four parameters that govern a forest's capacity: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.

`class_weight` is deliberately **not** in the grid, and there is a story there. It was originally, and the search took 45 minutes because it was correcting the same imbalance twice — SMOTE has already rebalanced every fold. Removing it halved the search with no measurable loss:

```python
# ml/train_random_forest.py
    ``class_weight`` is deliberately *not* in the grid. SMOTE has already
    rebalanced every training fold, and weighting the minority class on top of
    that corrects the same imbalance twice -- it doubled the search cost here for
    no measurable gain. A dataset where resampling is inappropriate should turn
    SMOTE off and reinstate the weighting, rather than run both.
```

The script also prints impurity importance next to the Shapley ranking — as a **diagnostic only**. Mean decrease in impurity is biased towards high-cardinality and continuous predictors, and it says nothing about the direction of an effect, which is why it is never allowed near a business user. But a predictor the forest splits on constantly that SHAP ranks nowhere is usually a leak or a correlated pair the data layer should have caught, and that is worth knowing at training time.

### 6.4 XGBoost

`ml/train_xgboost.py`. The parameter space is too large for exhaustive search, so it is tuned by Bayesian optimisation (Optuna, TPE sampler) over learning rate, depth, subsampling ratios, both regularisation terms, and instance weighting.

Two design points.

**`scale_pos_weight` is bounded above by the observed imbalance ratio, capped at 4.** SMOTE has already rebalanced the folds; weighting the positive class by the *full* ratio on top of that double-counts the correction and pushes the model into recall at any price.

**The number of boosting rounds is fixed by early stopping on a held-out validation fold**, not by the grid. This is the fiddliest code in the pipeline, because early stopping needs a validation set *in the transformed feature space*, and the transform lives inside the pipeline:

```python
# ml/train_xgboost.py
def finalise(pipeline, data, y_train, args):
    """Fix the number of boosting rounds by early stopping on a validation fold.

    A fold is carved out of the *training* partition; the preprocessing and the
    sampler are fitted on the remainder only, so the validation fold is scored by
    a model that has never seen it, directly or through a synthetic neighbour.
    """
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        data.X_train, y_train, test_size=VALIDATION_FRACTION,
        stratify=y_train, random_state=args.seed,
    )
    probe = clone(pipeline)
    preprocessor = probe.named_steps[PREPROCESS_STEP]
    variance = probe.named_steps[VARIANCE_STEP]
    matrix_fit = variance.fit_transform(preprocessor.fit_transform(X_fit))
    matrix_validation = variance.transform(preprocessor.transform(X_validation))

    sampler = probe.named_steps.get(SAMPLER_STEP)
    if sampler is not None and sampler != "passthrough":
        matrix_fit, y_fit = sampler.fit_resample(matrix_fit, y_fit)
    ...
```

Note the ordering: the preprocessor is `fit` on the *fit* portion only and merely `transform`s the validation fold, and SMOTE touches the fit portion only. The validation fold is scored by a model that has never seen it — not directly, and not through a synthetic neighbour interpolated from it.

The script closes with a feature-importance analysis that contrasts XGBoost's native `gain` with mean |SHAP|, and explains in the log why only one of them reaches the dashboard: gain is global, unsigned and model-internal — it cannot tell a retention manager whether a driver pushed *this* customer's risk up or down.

### 6.5 Calibration

The decision layer consumes probabilities, not labels. A miscalibrated `0.8` is operationally misleading, and tree ensembles are famously miscalibrated. So every model is calibrated with isotonic regression fitted on **out-of-fold** probabilities from the training partition:

```python
# ml/calibration.py
def cross_val_probabilities(estimator, X, y, folds=CALIBRATION_FOLDS, seed=RANDOM_SEED):
    """Out-of-fold probabilities for the training partition.

    The estimator is cloned and refitted inside each fold, so no record is ever
    scored by a model that saw it -- and because the sampler lives inside the
    pipeline, SMOTE runs within the fold rather than across the split. These
    probabilities are the honest validation signal used both to fit the
    calibrators and to choose the operating threshold; the test partition is not
    touched by either decision.
    """
```

Isotonic was chosen over Platt scaling for a specific reason: it is **monotone**. A monotone map cannot reorder customers, which means it cannot reorder the Shapley attributions either — the explanation still describes the model that produced the score. (It also means calibration cannot rescue a bad ranking, which is the right trade: if the ranking is bad, fix the model.)

The artifact stores plain `IsotonicRegression` objects and a small helper applies them, so the pickle contains **only** stock scikit-learn objects. The Django service can unpickle a model without importing any bespoke estimator class — one less thing to break in production.

On the banking XGBoost model this moved the Brier score from **0.1378 → 0.1027** on validation folds. That is a large improvement in the one number the decision layer actually depends on.

### 6.6 The operating threshold is a business decision

Nothing in this system defaults to 0.5.

```python
# ml/evaluation.py
def select_threshold(y_true, proba_positive, min_precision=MIN_PRECISION) -> ThresholdChoice:
    """Maximise F1 subject to ``precision >= min_precision``.

    If no threshold clears the floor, the unconstrained best-F1 threshold is
    returned and flagged, so that the shortfall is visible in the model report
    rather than silently absorbed.
    """
```

The threshold is chosen on the validation folds — never the test set — to maximise F1 **subject to a precision floor** agreed with whoever owns the retention budget. Precision is what decides how much of that budget is spent on customers who were never going to leave, which makes the threshold a commercial decision that merely looks technical.

The banking XGBoost model landed on **0.3288** with a 0.60 precision floor, and the floor was met. When it *cannot* be met, the code does not quietly lower its standards: it falls back to unconstrained best-F1, flags `satisfied_floor: False`, and the model card renders that as a red badge. A system that hides its own shortfalls is worse than no system.

### 6.7 Evaluation metrics

Accuracy alone is uninformative under imbalance — a classifier that never predicts churn is 79% accurate on the banking data. So `ml/evaluation.py` reports:

- **Precision / recall / F1** — wasted budget, missed revenue, and their balance.
- **ROC-AUC** — threshold-free discrimination.
- **MCC** — reported *beside* AUC because ROC-AUC flatters models under heavy imbalance; MCC is the more conservative summary of the same confusion matrix.
- **Brier score** — because the decision layer consumes probabilities.
- **Campaign value** — the confusion matrix valued at the operating threshold, in the spirit of the expected-maximum-profit criterion.

The multiclass path is implemented throughout (macro averages, one-vs-rest AUC, a multiclass Brier over the one-hot truth) even though both current datasets are binary, because the demand-forecasting variant of this system needs it and retrofitting it later would mean touching every metric.

### 6.8 Choosing between the three

`ml/train_all.py` prints a **dual-axis** comparison — discrimination and calibration on one axis, explanation stability and fidelity on the other — and then recommends a model under a deliberately conservative rule:

```python
# ml/train_all.py
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
```

That third clause is the one that matters, and it is the paper's argument compiled into code.

---

## 7. The explanation layer

### 7.1 SHAP, and the shapes nobody warns you about

`ml/explain.py` computes Shapley values for every scored customer — exactly and in polynomial time for the tree ensembles via `TreeExplainer`, and via the linear explainer for logistic regression.

The one genuinely irritating engineering problem here is that SHAP returns a different array shape depending on model family and library version: `(n, f)`, `(n, f, k)`, `(k, n, f)`, or a *list* of `(n, f)` arrays. Getting this wrong doesn't crash — it silently transposes your explanation. So the shape is normalised once, in one place, and never thought about again:

```python
# ml/explain.py
def shap_matrix(explainer, matrix, class_index, n_classes) -> np.ndarray:
    """Return the (rows x features) attribution matrix for one class.

    SHAP returns different shapes for different model families and versions --
    ``(n, f)``, ``(n, f, k)``, ``(k, n, f)`` or a list of ``(n, f)`` arrays --
    so the shape is normalised here once and never thought about again.
    """
    ...
    if values.ndim == 2:
        # Binary tree/linear models emit attributions for the positive class
        # only; the negative class is their mirror image.
        return values if class_index == 1 or n_classes <= 2 else -values
```

The background sample used as the explainer's reference distribution is **stored in the artifact**, so explanations recomputed at scoring time in the web app use the same reference as the ones audited at training time.

### 7.2 Auditing SHAP against LIME

Post-hoc explanations can be fragile. SHAP assumes feature independence in its common approximations, which distributes credit oddly across correlated features; LIME is unstable across re-runs of its own sampling. Neither is self-certifying.

So every model's explanations are **audited against an independent explainer** on a fixed sample of 100 test customers, stratified across the risk spectrum (auditing only the confident predictions would flatter everybody). For each customer we take the top five drivers from each method and measure:

- **overlap** — how many of the five they agree on;
- **ordering agreement** — a normalised Spearman footrule over the shared drivers;
- **LIME's stability against itself** — the same audit re-run with a different seed.

And critically, the audit produces a list of *contested features* — the ones SHAP ranks highly and LIME routinely does not:

```python
# ml/explain.py
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
```

That list is stored in the artifact and carried all the way to the user interface, where a disputed driver arrives with a visible warning label rather than as settled fact.

### 7.3 What the audit actually found

This is the most interesting result in the project, and it is not the accuracy table.

| Model (banking) | SHAP∼LIME top-5 overlap | LIME self-stability | What LIME disputes |
|---|---|---|---|
| Logistic Regression | **84%** | 90% | `Geography` one-hot dummies only |
| Random Forest | 72% | 74% | `tenure_ratio`, `balance_to_salary`, `EstimatedSalary`, `Balance`, … |
| **XGBoost** | **66%** | 81% | `EstimatedSalary` (100%), `products_per_tenure` (100%), `CreditScore` (100%), `Balance` (96%), `Tenure` (94%) |

Read it slowly. **The most accurate model has the least corroborated explanations** — and the specific drivers LIME disputes (`Balance`, `Tenure`, `CreditScore`) are precisely the ones the intervention taxonomy leans on to justify actions (`churn_balance_risk`, `churn_short_tenure`). On the demo batch, that means **54% of XGBoost's recommendations rest on an attribution a second explainer would not endorse.**

The temptation is to tune that number away. We did not, because it is true, and because it is exactly the trade-off the dual-axis evaluation exists to surface: you can have the extra 0.10 of MCC, but you are buying it with explanation quality, and in a system whose entire purpose is to justify actions to humans, that is a real cost and not a rounding error.

(The logistic model's contested features are instructive too — they are *only* the `Geography` dummies, which is the textbook SHAP-with-correlated-one-hot-columns artefact: credit gets split across mutually exclusive dummies in a way LIME's local surrogate does not reproduce. That is a known limitation of the method, not a flaw in the model.)

---

## 8. The fairness audit

`ml/fairness.py` runs on the predictions the decision layer would actually act on, at the threshold it would act at — not on some other configuration that happens to look better.

For every declared sensitive attribute it reports, per group: the base churn rate, the **selection rate** (what share of the group the model would target), the **true positive rate** (what share of that group's real churners get caught — i.e. that group's share of the *protection* on offer), precision, and the four-fifths ratio.

On the banking XGBoost model it found this, and the model card publishes it (Figure 6):

| Attribute | Selection-rate ratio | TPR gap | Flag |
|---|---|---|---|
| Gender | 0.50 | 0.15 | below four-fifths |
| Country | 0.36 | 0.29 | below four-fifths |
| Age | **0.07** | **0.67** | below four-fifths |

Women are targeted at roughly twice the rate of men; the under-30 group is essentially never contacted, and only 23% of its actual churners are caught. `Age` is also the model's **single strongest driver**.

None of this is hidden, and none of it is automatically "fixed", because a gap is not automatically an injustice — it can faithfully reflect a real difference in churn between groups, and equalising it by fiat would withhold help from the group that most needs it. What the audit does is make the gap **impossible to ship without having looked at it**:

```python
# ml/fairness.py
    """...
    None of these numbers decides anything on its own. A selection-rate gap can be a
    faithful reflection of a real difference in churn between groups, and equalising
    it by fiat would deny help to the group that needs it. What the audit does is
    make the gap impossible to ship without having looked at it.
    """
```

It also flags any demographic attribute that appears among the model's leading drivers as a **proxy candidate** — a finding that demands an independent business justification or the feature comes out.

---

## 9. The decision layer

This is the step that distinguishes the system from a well-tuned classifier with a chart on top.

### 9.1 The intervention taxonomy

`decision/intervention_rules.json` maps churn drivers to retention actions. It is a **data file, not code** — and that is a deliberate, load-bearing decision. Retention offers change monthly; developers do not deploy monthly. A retention manager can edit this file, and the system picks up the change on the next scoring run, with no redeployment and nobody's sprint interrupted.

A rule looks like this:

```json
{
  "id": "churn_service_failure",
  "applies_to": ["1"],
  "match": { "category": "engagement" },
  "intervention": "Priority service recovery call within 48 hours",
  "owner": "Service recovery",
  "weight": 0.95,
  "rationale": "Complaint-driven risk decays fastest of all: a customer with an unresolved complaint is deciding now, and an offer arriving before the apology insults them."
}
```

The `rationale` is not decoration — it is rendered in the interface under "Why this action", and it is copied into the audit log at scoring time rather than looked up later, because the rule table is editable and the reason a customer was routed somewhere in March must still be readable in September.

The file's own `notes` section is candid about what it is:

> *The judgement encoded here — that a given driver is best answered by a given offer — rests on the retention-marketing literature and on face validity, not on experimental evidence from the firms that would use it. Different sectors price interventions differently, and a mapping that pays for a bank may be loss-making for a streaming service. Editability transfers that burden to the adopting firm rather than discharging it.*

### 9.2 The rule engine

`ml/decision.py` walks a customer's drivers strongest-first, finds the first rule that matches, and produces the recommendation. Rules are ordered specific-before-general, and every actionable category terminates in a rule for both classes — a predicted stayer never falls through to "manual review", because a queue that asks a human to look at customers who are not leaving is a queue nobody reads.

### 9.3 The demographic invariant

Here is the rule that matters most in the entire codebase.

A customer's age or nationality may legitimately be the strongest driver of their churn estimate. Suppressing that from the explanation would make the explanation a lie. But **the business may not act on it**. So the matching loop runs over *actionable* drivers only:

```python
# ml/decision.py
    # A demographic driver may be shown -- suppressing it would make the
    # explanation a lie -- but it may never generate an offer. Rules are matched
    # against the actionable drivers only, so that a customer whose risk is
    # dominated by age or nationality still receives a recommendation grounded in
    # something the business can legitimately change, and the dominance itself is
    # raised as a flag rather than buried.
    actionable = [driver for driver in drivers if not _protected(driver)]
    protected = [driver for driver in drivers if _protected(driver)]

    matched = []
    for driver in actionable:
        for rule in rules:
            if _applies_to(rule, predicted_label) and _matches(rule, driver):
                matched.append((rule, driver))
                break
```

The behaviour that falls out of this:

- **Demographic driver is strongest, an actionable one follows** → the offer is built from the strongest *actionable* driver, and a `demographic_driver_dominant` flag is raised with a visible caveat.
- **Only demographic drivers** → the system refuses to make an offer at all and routes the customer to a human being.

And the guard is belt-and-braces — `_protected()` recomputes the flag from the feature name if a caller forgot to set it, "so that a rule table can never be tricked into making an offer on a protected attribute by a caller that forgot to set it."

This invariant has a test of its own (`test_no_offer_is_ever_made_on_a_demographic_driver`), because an ethical guarantee that isn't tested is an ethical aspiration.

You can see it working on live data in Figure 5: customer 15618225's two strongest drivers are `Age` and `Country: Germany`, both tagged **"not used for offers"** — and the recommendation reaches past both to "Products held" and offers a cross-sell.

### 9.4 Priority: risk weighted by value

Targeting the customers whose retention creates the most value is *not* the same as targeting those most likely to leave, and a queue sorted on probability alone quietly assumes it is. So:

```
priority_score = probability × rule_weight × value_multiplier
```

The value multiplier is the customer's value (account balance, monthly fee — whatever the dataset offers) expressed as a multiple of the **training-set median**, which is recorded in the artifact. That indirection is what lets one rule table serve a dataset denominated in six-figure account balances and one denominated in £8.99 subscriptions without either of them needing its own copy.

---

## 10. The model registry

`ml/registry.py`. Every trained model is stored with its data snapshot (SHA-256 of the source CSV), hyperparameters, evaluation report, explanation audit and fairness audit. Every score the dashboard serves records the model version that produced it.

This bookkeeping is unglamorous and it is the difference between a demo and a system. When a retention decision is challenged six months later — by a manager, an auditor, or the customer — you have to be able to reproduce the prediction, the explanation *and* the recommendation exactly as they stood on the day. A score without its model version cannot do that.

The manifest is written atomically, because a reader must never see a half-written file:

```python
# ml/registry.py
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    temporary.replace(MANIFEST_PATH)  # atomic: a reader never sees a half-file
```

Old versions are kept, never deleted. They are the evidence behind decisions already taken.

---

## 11. The web application

### 11.1 Settings and security

`bdss_project/bdss_project/settings.py`. Development defaults are convenient; production defaults are strict, and the switch is a single environment variable:

```python
if not DEBUG:
    # Customer data travels over this connection; it does not travel in clear.
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 31536000
    ...
```

One setting deserves special mention:

```python
#: Retain the raw uploaded feature values alongside each logged prediction.
#: Turn this off where data minimisation requires that only the score and the
#: drivers are retained; the audit trail then keeps the reasons but not the data.
BDSS_LOG_INPUT_DATA = os.environ.get("BDSS_LOG_INPUT_DATA", "1") == "1"
```

Under GDPR's data-minimisation principle you may not be entitled to keep the personal data you scored. Setting this to `0` keeps the score, the drivers and the recommendation — the full accountability trail — while discarding the personal data behind it. That is a genuine tension (auditability *versus* minimisation) and the system resolves it by making it a deliberate, documented choice rather than an accident.

### 11.2 Authentication

Nothing in this system is anonymous. Every screen requires a signed-in user, and every prediction served is attributed to the person who requested it.

![Login page](figures/fig01-login.png)

**Figure 1 — The sign-in screen (`/accounts/login/`).** Standard Django auth, with `LoginView` and a template at `dashboard/templates/registration/login.html`. The copy is deliberate: it tells the user *why* there is a login, because "this system holds customer-level risk estimates and the reasons behind them" is a better explanation of a password box than "Sign in".

The API enforces the same rule (`IsAuthenticated` by default in `REST_FRAMEWORK`), so an unauthenticated `POST /api/predict/` returns 403, not a prediction. There is a test for it.

### 11.3 `ml_engine/predictor.py` — the inference path

This module is the join between the two halves of the system. It loads registry artifacts, scores customers, explains each score with the *same* Shapley machinery the training run audited, and passes the drivers through the *same* intervention taxonomy.

Three properties are load-bearing, and the module's docstring names them as such: the probability served is the **calibrated** one; the label served is the one the **registered operating threshold** implies (never the 0.5 arg-max); and the model version is recorded with every score.

Models are cached in-process and reloaded when the artifact changes on disk, so retraining does not require restarting the web server. And when a requested model isn't there, it raises rather than substituting:

```python
# bdss_project/ml_engine/predictor.py
class ModelUnavailable(RuntimeError):
    """No model is registered under the requested key.

    Raised rather than silently falling back to another model: a retention
    decision must never be attributed to a model that did not make it.
    """
```

A subtle point in `predict()`: the risk figure is *always* the probability of the churn class, but "confidence" is the probability of whichever class the model actually assigned. Those are different questions, and conflating them is how a dashboard ends up telling a manager that a customer is "96% confident" when it means "96% likely to stay". They are reported separately.

The schema reconciliation is deliberately forgiving in one direction and honest in the other: unknown columns in the upload are ignored (so a raw CRM export works), missing columns are imputed by the pipeline *and reported back* in the metadata, because a silently imputed column is the difference between a score and a guess. Derived features are not counted as "missing" — `_derivable()` works out what `engineer_features` would manufacture from what was supplied, so nobody goes hunting for a `balance_to_salary` column that never existed in any CRM.

### 11.4 `ml_engine/utils.py` — intake, summaries, charts

CSV intake with human-readable failures ("The file has a header row but no customers in it"), batch summaries, CSV export, and the three charts.

The charts are rendered **server-side with matplotlib and embedded as base64 PNGs**. This was a considered choice over Plotly or a JavaScript charting library: it keeps the dashboard free of any external script, font or CDN request, which matters when the page is displaying customer-level risk inside a corporate network. Nothing on this page phones anywhere.

Colour follows three rules, and only these:

```python
# bdss_project/ml_engine/utils.py
#: Status scale for the risk bands. Validated against the page surface: all three
#: clear 3:1 contrast. Their worst adjacent CVD separation is at the floor, so
#: they are only ever used with a text label alongside.
BAND_COLOURS = {"high": "#b42318", "medium": "#b54708", "low": "#067647"}

#: Categorical scale for the feature categories, in fixed order. Assigned by
#: category, never by rank, so a filter that removes a category does not repaint
#: the survivors.
CATEGORY_COLOURS = {
    "behavioural": "#2a78d6", "transactional": "#1baf7a", "contractual": "#eda100",
    "engagement": "#008300", "demographic": "#4a3aa7", "uncategorised": "#898781",
}
```

Both palettes were run through a colour-blindness validator. The risk bands (red / amber / green) sit at the *floor* of adjacent CVD separation — red and amber are close under deuteranopia — which is why **every band is written out in text beside the colour**. Colour never carries meaning on its own anywhere in this interface.

### 11.5 The audit trail (`dashboard/models.py`)

Two tables. `PredictionBatch` is one scored customer list. `Prediction` is one customer: what was scored, what scored it (model key, label, **version**, threshold), what it said (label, probability, confidence, band), *why* it said it (`top_drivers` as JSON), what to do about it (recommendation, owner, priority, rule id, rationale, caveats) — and what the human did about it.

That last group is the point:

```python
# bdss_project/dashboard/models.py
    """...
    The override fields are the other half of the same commitment. The system is
    advisory: a retention manager can reject any recommendation, and when they do,
    the rejection is recorded with its reason. That log serves two purposes. It feeds
    the next model review, and it is the only way to detect automation bias -- the
    tendency to defer to the system precisely because it is a system. A queue with no
    overrides is not a sign of a good model.
    """
```

The override *reason* is mandatory. An override with no reason tells the next model review nothing, and telling the next model review something is the entire purpose of capturing it.

### 11.6 Overview

![Overview screen](figures/fig02-overview.png)

**Figure 2 — The overview (`/`).** Four stat tiles, the models the registry can currently serve, recent batches, and the latest predictions.

Two things on this screen are unusual and intentional.

First, the model table shows a column headed **SHAP∼LIME** next to F1 and recall. Most model pickers show you accuracy. This one shows you accuracy *and* whether the model's explanations survived an independent audit, because in this system a model whose explanations cannot be corroborated should not be driving offers however well it ranks. The dual-axis evaluation is not buried in a report; it is in the dropdown.

Second, look at the fourth tile: **"Overrides logged: 0.0% — Nobody has disagreed with the model yet."** That copy is deliberately uncomfortable. An override rate of zero is not a triumph; it is the signature of automation bias, and the interface says so on the home page rather than waiting for someone to go looking.

### 11.7 Upload

![Upload screen](figures/fig03-upload.png)

**Figure 3 — Scoring a customer list (`/upload/`).** A file input, a model dropdown, and a filter checkbox.

The model dropdown is built from the registry at request time (`dashboard/forms.py`), not from a hard-coded list — so a model trained this morning is selectable this afternoon, and a withdrawn model cannot be chosen at all. Each option carries its F1 and its version. The default is the strongest F1 among registered models, but the *choice* stays the user's: the framework recommends, it does not lock.

The right-hand panel is a small piece of honesty engineering. Each model knows only its own domain's columns, so the panel says so plainly — "score a banking list with a banking model" — and shows each model's operating threshold, with a note explaining why it isn't 0.5.

### 11.8 Results — the retention queue

![Results summary](figures/fig04-results-summary.png)

**Figure 4 — Batch summary (`/batch/<id>/`).** 400 customers scored by XGBoost v20260714T163911Z, operating threshold 0.329, taxonomy v1.0 — all stated in the sub-heading, because a page of risk estimates with no provenance is a rumour.

The three charts answer three different questions. The **risk distribution** shows where the book sits and where the model draws its line (the dashed threshold marker). The **priority chart** shows the shape of the work: 29 customers on fire, 33 for this cycle, 338 to monitor. The **driver-category chart** is the campaign-level read — on this batch, "Contract and products" carries 43% of the total attribution, which tells a manager that the answer here is probably a structural change to contract terms rather than 85 individual phone calls.

![Results queue](figures/fig05-results-queue.png)

**Figure 5 — The retention queue.** This is the product. Every row is a customer, a risk band, three named drivers, an action, an owner and a priority.

Note what is happening in these rows:

- Customer **15618225** — strongest drivers are `Age` and `Country: Germany`, both tagged **"not used for offers"**. The recommendation reaches past both to `Products held` and offers a cross-sell. The amber caveat explains exactly that, in words: *"The strongest driver of this estimate is Age, a demographic attribute. No offer is made on that basis."* This is §9.3 in action.
- Customer **15587597** — carries *two* caveats: the demographic one, and *"This action rests on Years as a customer, a driver on which SHAP and LIME disagreed during the explanation audit. Confirm before acting."* This is §7.3 reaching the person who has to make the call.
- The **disputed** chips mark contested drivers wherever they are displayed, even when the recommendation does not rest on them.

That distinction — *mark* a disputed driver wherever it is shown, but only *caveat the action* when the action depends on it — was a late correction. The first implementation raised a caveat whenever any displayed driver was contested, and 72% of the queue lit up. A warning that fires on three customers in four is a warning nobody reads. Now the row-level warning means something specific, and the inline chip carries the rest.

Every row also expands into an override form. The legend on it reads: *"You know the customer. The model does not."*

### 11.9 Model cards

![Model card](figures/fig06-model-card-bank.png)

**Figure 6 — The model card (`/model/xgboost/bank_churn/`).** Everything about a registered model on one page: headline metrics, the operating point and *how it was chosen*, the campaign value, the confusion matrix, the leading drivers, the explanation audit (including the six drivers the two explainers dispute), the fairness audit with its flags, and full provenance — training time, seed, row counts, the SHA-256 of the source data, and every hyperparameter.

This screen exists because a model that cannot be interrogated cannot be trusted with a retention budget. Note that it publishes the model's *failures* as prominently as its successes: the four-fifths flags, the TPR gaps, the contested drivers, and a plain-English note that "a gap is not automatically an injustice… what the audit does is make the gap impossible to ship without having looked at it."

![Netflix model card](figures/fig07-model-card-netflix.png)

**Figure 7 — The streaming model card, with a health warning.** The streaming models post ROC-AUC 0.9999 and MCC 0.984. Those are not results; they are a warning, and the card says so before it says anything else.

The label in that dataset is synthetic: a depth-5 decision tree recovers 93% of it from three columns, in a clean threshold rule over `avg_watch_time_per_day`, `last_login_days` and `watch_hours`. The models are rediscovering the generator that wrote the file, not predicting cancellations.

The mechanism here is worth noting for reviewers. The dataset *description* is read live from `ml/config.py` at render time rather than from the frozen artifact:

```python
# bdss_project/dashboard/views.py
    # The dataset description is read from the live configuration rather than from
    # the artefact. What we know about a dataset can change after a model is
    # trained on it -- a leak found, a label discovered to be synthetic -- and that
    # discovery has to reach the person reading the metrics, not sit frozen inside
    # a pickle written before anyone knew.
```

We learned that the label was synthetic *after* the models were trained. A warning that could only be attached at training time would have been useless.

### 11.10 Taxonomy and registry

![Taxonomy](figures/fig08-taxonomy.png)

**Figure 8 — The published intervention taxonomy (`/taxonomy/`).** Every rule, its trigger, its action, its owner, its weight, its rationale — and how many times it has fired.

Publishing this is a fairness safeguard as much as a usability one. If offers flow only to the customers a model flags, the groups it never flags quietly receive worse treatment, and a rule table nobody outside the team can read makes that invisible. The "Applied" column is a live diagnostic: a rule that never fires is dead weight, and a rule that fires for everything is not discriminating.

![Registry](figures/fig09-registry.png)

**Figure 9 — The model registry (`/registry/`).** Every version ever trained, with F1, MCC, Brier, the explanation-audit score, and how many customers it has scored. The dashboard serves the newest version of each model; older versions are kept because they are the evidence behind decisions already taken.

### 11.11 The REST API

Two endpoints (`dashboard/api.py`):

```
GET  /api/models/    the models the registry can currently serve
POST /api/predict/   score customers, return the recommended action
```

```bash
curl -u admin:retention2026 -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"model": "xgboost|bank_churn",
       "records": [{"CreditScore": 608, "Geography": "Germany", "Gender": "Male",
                    "Age": 58, "Tenure": 1, "Balance": 125510.82, "NumOfProducts": 1,
                    "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 112542.58}]}'
```

The response carries the probability, the drivers, the recommendation, the owner, the priority and the model version — because a CRM that consumes the score without the reasons reintroduces the exact problem this system was built to solve. It also carries a `notice` field restating that Shapley attributions describe the model, not the customer.

Two design points. A model is identified by `model_key|dataset`, not just `xgboost` — "a random forest trained on banking data and one trained on streaming data are different models with different feature schemas; collapsing them onto the label 'Random Forest' is how the wrong model ends up scoring the wrong file." And **API predictions are written to the same audit trail as dashboard ones**: a prediction made over HTTP is still a prediction about a real person, and it earns the same accountability.

### 11.12 The admin log

![Admin log](figures/fig10-admin-log.png)

**Figure 10 — The prediction log (`/admin/dashboard/prediction/`).** Searchable, filterable by priority, band, source, model version and override status.

Predictions are **read-only** here — `has_add_permission` and `has_change_permission` both return `False`. The log is evidence, and evidence that can be edited after the fact is not evidence. Overrides are made through the dashboard, where they are attributed and reasoned; the admin exists to search and inspect, not to rewrite.

---

## 12. Testing

`bdss_project/dashboard/tests.py` — fourteen tests. They are not unit tests of scikit-learn. They test the things that would quietly break a retention decision **without ever raising an exception**:

| Test | What it protects |
|---|---|
| `test_every_screen_requires_a_signed_in_user` | No screen showing customer risk is reachable anonymously |
| `test_api_rejects_anonymous_callers` | Same, for the API |
| `test_no_offer_is_ever_made_on_a_demographic_driver` | **The ethical invariant of §9.3** |
| `test_upload_creates_a_batch_and_logs_every_prediction` | Every served score carries its model version |
| `test_an_override_is_recorded_against_the_person_who_made_it` | Human accountability |
| `test_override_without_a_reason_is_refused` | An unreasoned override is useless to the next review |
| `test_export_carries_the_reasons_and_the_model_version` | A scored list never leaves the building without its reasons |
| `test_api_predictions_are_written_to_the_same_audit_trail` | HTTP predictions are not second-class |
| `test_an_unknown_model_is_refused_rather_than_substituted` | No decision is attributed to a model that did not make it |
| `test_a_junk_upload_fails_with_an_explanation_not_a_stack_trace` | A retention manager can act on the error |

Tests requiring a trained model skip cleanly on a fresh checkout (`@requires_model`) and run for real once `ml.train_all` has been run — so the suite is green on a clean clone and meaningful after training.

---

## 13. Results, and what they mean

**Banking (20,000-row stratified sample of 165,034; 21.2% churn). Judge the system on these.**

| Model | Precision | Recall | F1 | ROC-AUC | MCC | Brier | SHAP∼LIME | Train time |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.699 | 0.441 | 0.541 | 0.831 | 0.468 | 0.117 | **84%** | 31 s |
| Random Forest | 0.614 | 0.677 | 0.644 | 0.881 | 0.543 | 0.101 | 72% | 22 min |
| **XGBoost** | 0.624 | 0.713 | **0.665** | **0.883** | **0.570** | **0.099** | 66% | 3 min |

XGBoost is recommended: it beats the transparent baseline by 0.10 MCC, comfortably clearing the one-point bar in §6.8. At its operating threshold (0.329) on the 4,000-row test partition it catches **603 of 846** churners, misses 243, and contacts 364 people who were never going to leave — a campaign value of **+1.56 per customer scored** under the placeholder economics.

But read the two right-hand columns together. The accuracy ladder runs one way and the explanation ladder runs the other. That is not an artefact of this dataset; it is the central tension of the whole field, and the reason this system measures both.

**Streaming.** Ignore the numbers (§11.9, Figure 7). The label is synthetic. The dataset is retained because it exercises every layer of the framework on a second domain with a completely different feature vocabulary — which is what the cross-domain design needs from it — but it is not evidence of anything about real churn.

**The fairness findings** (§8) are real properties of the banking data, and they are published rather than patched.

---

## 14. Limitations and future work

Stated plainly, because a manual that only lists strengths is marketing.

1. **Two domains, not three.** The e-commerce arm is unlabelled (§4.2). Cross-domain transfer is evidenced across banking and streaming only.
2. **One of those two domains is synthetic.** So the strongest real evidence rests on a single dataset. This is the biggest single weakness in the evaluation, and the honest fix is a second *real* labelled churn dataset.
3. **Shapley attributions describe the model, not the world.** The actions the taxonomy triggers are motivated correlations, not established causes. The system flags customers and reasons; it cannot promise that treating the reason will avert the churn. This is stated in the footer of every page for a reason.
4. **The taxonomy is untested.** It encodes a judgement resting on the retention-marketing literature and face validity, not on experiment. Editability transfers that burden to the adopting firm rather than discharging it. Its end-to-end value remains a hypothesis until measured against realised retention outcomes, ideally against a holdout control group.
5. **The explanation audit is evidence of robustness, not proof of fidelity.** Agreement between two post-hoc methods means they agree, not that either is right.
6. **The dashboard is designed against personas, not validated with users.** A study with practising retention staff would test whether the explanations actually change decisions.
7. **The banking model was trained on a 20,000-row subsample** for tractable iteration. A production run should use all 165,034 rows; the code supports it (drop `--max-rows`), it simply takes hours.

**Where we would go next, in order:**

1. **Uplift modelling.** The natural methodological extension: estimate the *effect* of an intervention on each customer rather than their risk, converting the decision layer from risk-based to effect-based targeting. Right now the system targets who is leaving, not who can be persuaded to stay — and those are different people.
2. **Replace the placeholder economics** with the adopting firm's real numbers. This is a one-line change that moves the operating threshold and therefore the entire campaign.
3. **A field trial against a control group.** Everything in this system is machinery for making a retention decision defensible. Whether it makes retention *better* is a question only a trial can answer.

---

## 15. Glossary

**Brier score** — Mean squared error between predicted probability and the truth. Measures *calibration*: whether a stated 0.8 really means 80%. Lower is better. Matters here because the decision layer consumes probabilities, not labels.

**Calibration** — Adjusting a model's raw scores so they behave like real probabilities. Tree ensembles are typically overconfident; we correct this with isotonic regression on out-of-fold predictions (§6.5).

**Contested feature** — A driver that SHAP ranks among a customer's top five but LIME routinely does not. Surfaced by the explanation audit, marked "disputed" in the interface (§7.2).

**Four-fifths rule** — A convention: if one group's selection rate is below 80% of the highest group's, investigate. A trigger for review, not a verdict (§8).

**Isotonic regression** — A non-parametric, *monotone* calibration map. Monotone matters: it cannot reorder customers, so it cannot reorder their Shapley attributions.

**LIME** (Local Interpretable Model-agnostic Explanations) — Fits a simple linear model in the neighbourhood of one prediction to explain it. Cheap and intuitive; unstable across re-runs. Used here as the *independent auditor* of SHAP, not as the primary explainer.

**MCC** (Matthews correlation coefficient) — A single-number summary of the confusion matrix that stays honest under class imbalance, where accuracy and ROC-AUC flatter. The primary model-selection metric here.

**Operating threshold** — The probability above which a customer is treated as a churner. Chosen on validation folds to maximise F1 subject to a precision floor — never left at 0.5 (§6.6).

**Out-of-fold prediction** — A prediction made for a record by a model that was not trained on it. The only honest validation signal, and what both calibration and threshold selection are fitted on.

**Priority score** — `probability × rule_weight × value_multiplier`. What sorts the work queue, so that a near-certain churner worth very little does not outrank a slightly less certain one worth a great deal (§9.4).

**Protected / demographic category** — Age, gender, geography, device. Permitted as predictors so the fairness audit can test them; forbidden from generating an offer (§9.3).

**SHAP** (SHapley Additive exPlanations) — Assigns each feature a contribution to an individual prediction, grounded in cooperative game theory. Exact and fast for tree ensembles via `TreeExplainer`. The primary explanation method here.

**SMOTE** (Synthetic Minority Over-sampling TEchnique) — Invents synthetic minority-class examples by interpolating between real ones, to counter class imbalance. Must be applied *inside* each cross-validation fold or it leaks (§4.4).

**Selection rate** — The share of a group the model would target. Compared across groups by the fairness audit.

**True positive rate (per group)** — The share of a group's *actual* churners the model catches. Effectively that group's share of the protection the system offers — which is why a gap in it is an equity problem, not just a statistics problem.

---

## 16. Conclusion and recommendations

The argument this codebase makes, in one sentence: **the useful frontier in churn prediction is no longer accuracy, but the distance between a probability and a decision** — and that distance is closed with engineering, not with a better model.

Concretely, the system closes it with a five-layer pipeline in which every layer is answerable to the next. The data layer refuses to leak (SMOTE inside the fold; correlated predictors dropped on training data only). The model layer refuses to flatter itself (a threshold chosen on validation folds against a precision floor; MCC beside AUC; calibration measured, not assumed). The explanation layer refuses to be taken on trust (SHAP audited against LIME, with the disputes carried all the way to the user's screen). The decision layer refuses to launder a demographic attribute into an offer. And the whole thing refuses to serve a score without saying which model version produced it.

The engineering decisions we would defend most strongly, if challenged:

1. **The web tier imports the pipeline rather than reimplementing it** (§3.3). Everything else in this system depends on the served explanation being the trained explanation.
2. **The intervention taxonomy is a data file** (§9.1). Retention offers change monthly; deployments do not.
3. **The demographic invariant is enforced in code and covered by a test** (§9.3). An ethical guarantee that isn't tested is an ethical aspiration.
4. **The uncomfortable numbers are published, not patched** — the fairness gaps, the contested drivers, the synthetic-label warning, the zero override rate. A decision-support system that hides its own weaknesses is not a decision-support system; it is a liability with a nice interface.

**For whoever picks this up next**, in priority order: replace the placeholder economics with real ones; get a second *real* labelled dataset so the cross-domain claim stands on more than one leg; take the explanation-audit finding seriously and decide, with the business, whether XGBoost's extra 0.10 of MCC is worth the 18-point drop in explanation corroboration (that is a commercial conversation, not a technical one, and the model card gives you the numbers to have it); and then run a field trial, because everything here is machinery for making a decision defensible, and only a trial can tell you whether it makes the decision *better*.

---

## Appendix A: File map

| Path | Lines | Purpose |
|---|---|---|
| `ml/config.py` | 253 | Paths, seeds, protocol constants, dataset registry, unit economics |
| `ml/data_prep.py` | 399 | Cleaning, encoding, scaling, splitting, SMOTE pipeline |
| `ml/features.py` | 269 | Five categories, derived predictors, business-readable names |
| `ml/calibration.py` | 100 | Out-of-fold probabilities, isotonic calibrators |
| `ml/evaluation.py` | 328 | Metric suite, threshold selection, campaign value |
| `ml/explain.py` | 383 | SHAP, LIME audit, contested-feature detection |
| `ml/fairness.py` | 208 | Per-group error rates, four-fifths check, proxy detection |
| `ml/decision.py` | 297 | Rule engine, demographic invariant, priority scoring |
| `ml/registry.py` | 200 | Versioned artifact registry with atomic manifest |
| `ml/train_common.py` | 599 | The shared seven-stage training protocol |
| `ml/train_logistic_regression.py` | 152 | Baseline + log-odds report |
| `ml/train_random_forest.py` | 162 | Grid search + impurity-importance diagnostic |
| `ml/train_xgboost.py` | 279 | Optuna search, early stopping, gain-vs-SHAP analysis |
| `ml/train_all.py` | 201 | Dual-axis comparison and model recommendation |
| `decision/intervention_rules.json` | 228 | The editable taxonomy (18 rules) |
| `bdss_project/bdss_project/settings.py` | 219 | Settings, security, BDSS configuration |
| `bdss_project/dashboard/models.py` | 184 | `PredictionBatch`, `Prediction`, override fields |
| `bdss_project/dashboard/views.py` | 473 | Overview, upload, results, model card, taxonomy, registry |
| `bdss_project/dashboard/api.py` | 169 | `/api/predict/`, `/api/models/` |
| `bdss_project/dashboard/tests.py` | 284 | Fourteen tests |
| `bdss_project/ml_engine/predictor.py` | 388 | Model cache, inference, explanation, recommendation |
| `bdss_project/ml_engine/utils.py` | 439 | CSV intake, summaries, CSV export, three charts |

## Appendix B: Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | dev key | **Must** be set in production |
| `DJANGO_DEBUG` | `1` | `0` enables HSTS, secure cookies, SSL redirect |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated |
| `BDSS_MODELS_DIR` | `models/` | Where the registry lives |
| `BDSS_RULES_PATH` | `decision/intervention_rules.json` | The editable taxonomy |
| `BDSS_MAX_UPLOAD_ROWS` | `20000` | Upload cap |
| `BDSS_MAX_UPLOAD_BYTES` | `20 MB` | Upload cap |
| `BDSS_TOP_DRIVERS` | `3` | Drivers surfaced per customer |
| `BDSS_LOG_INPUT_DATA` | `1` | `0` retains reasons but not personal data |
| `BDSS_API_RATE` | `120/minute` | API throttle |
