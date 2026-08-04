# Retention BDSS — Interpretable Churn Prediction and Decision Support

A working implementation of the framework in *"Predicting Customer Churn in Subscription
Services Using Interpretable Machine Learning: A Business Decision-Support Framework for
Retention Management"* (Onoja, Ulster University).

The research argues that the useful frontier in churn prediction is no longer accuracy but
**the distance between a probability and a decision**. This repository closes that distance:
three classifiers of graded complexity, Shapley explanations audited against an independent
explainer, and an explicit taxonomy that turns each explanation into a prioritised retention
action with a named owner — delivered through a dashboard built for staff who know the
customers, not the models.

---

## Quick start

```bash
# 1. Environment  (Python 3.13; Windows paths shown)
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Train the models  (writes artefacts into models/ and reports into reports/)
python -m ml.train_all --dataset bank_churn --max-rows 20000
python -m ml.train_all --dataset netflix_churn

# 3. Web application
cd bdss_project
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open **http://127.0.0.1:8000/**.

**The account created during development is `admin` / `retention2026`.** It is a development
credential in a SQLite file; change it before this goes anywhere near a real customer list
(`python manage.py changepassword admin`).

Every command uses the virtualenv's interpreter. Without activating it, prefix with
`..\.venv\Scripts\python.exe` from inside `bdss_project/`.

---

## What you can do with it

| Route | What it is |
|---|---|
| `/` | Overview: what has been scored, which models are available, the override rate |
| `/upload/` | Upload a CSV customer list, pick a model, get a ranked retention queue |
| `/batch/<id>/` | The queue: risk band, three strongest drivers, recommended action, owner, priority |
| `/batch/<id>/export/` | CSV export for the CRM — carries the *reasons* and the model version, not just the score |
| `/taxonomy/` | The published rule table that maps drivers to interventions |
| `/registry/` and `/model/<key>/<dataset>/` | Model cards: metrics, calibration, explanation audit, fairness audit, provenance |
| `/api/predict/` | JSON endpoint returning probability, drivers, and recommendation |
| `/admin/` | The read-only audit trail of every prediction and override |

**Try it:** upload `datasets/Bank/train.csv` (or a slice of the 110k-row `datasets/Bank/test.csv`)
and score it with a banking model.

### The API

```bash
curl -u admin:retention2026 -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"model": "xgboost|bank_churn",
       "records": [{"CreditScore": 608, "Geography": "Germany", "Gender": "Male",
                    "Age": 58, "Tenure": 1, "Balance": 125510.82, "NumOfProducts": 1,
                    "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 112542.58}]}'
```

The response carries the calibrated probability, the ranked drivers in business language, the
recommended intervention, its owner and priority, and the model version that produced all of
it. A CRM that stores only the probability is throwing away the part of the answer this
framework exists to supply.

---

## Architecture

The five layers of Fig. 1 in the paper, each consuming the one above it:

```
  DATA LAYER          ml/data_prep.py      clean, encode, min-max scale, drop redundant
                                           predictors, 80/20 stratified split, SMOTE in-fold
        |
  FEATURE LAYER       ml/features.py       behavioural / transactional / contractual /
                                           engagement / demographic + derived ratios
        |
  MODEL LAYER         ml/train_*.py        logistic regression, random forest, XGBoost;
                                           10-fold CV, tuning, isotonic calibration,
                                           threshold chosen on validation folds
        |
  EXPLANATION LAYER   ml/explain.py        SHAP per customer (TreeExplainer),
                                           audited against LIME on 100 stratified records
        |
  DECISION LAYER      ml/decision.py       intervention taxonomy -> action, owner, priority
                      bdss_project/        dashboard, batch scoring, REST API, audit trail
```

**The web tier imports `ml/`; it does not reimplement it.** The feature vocabulary, the
calibration maths and the intervention taxonomy have exactly one definition, so what the
dashboard shows a retention manager is by construction what the training run measured and
registered. `bdss_project/bdss_project/settings.py` puts the repository root on `sys.path` to
make that import work.

### Files

```
ml/                          the pipeline
  config.py                  paths, seeds, protocol constants, dataset registry, unit economics
  data_prep.py               the data layer
  features.py                feature categories, derived predictors, business-readable names
  calibration.py             isotonic calibration on out-of-fold probabilities
  evaluation.py              precision, recall, F1, ROC-AUC, MCC, Brier, threshold, profit
  explain.py                 SHAP, and the LIME audit of SHAP
  fairness.py                per-group error rates, four-fifths check, proxy detection
  decision.py                the rule engine over the intervention taxonomy
  registry.py                versioned model registry
  train_common.py            the shared 7-stage training protocol
  train_logistic_regression.py / train_random_forest.py / train_xgboost.py
  train_all.py               trains the ladder and prints the dual-axis comparison

decision/intervention_rules.json    the editable taxonomy (Table II, made executable)
datasets/                           the source data
models/                             trained artefacts + registry.json
reports/                            evaluation reports and model comparisons

bdss_project/                dashboard/  (models, views, forms, API, templates, tests)
                             ml_engine/  (predictor.py, utils.py)

docs/paper_figures.py               rebuilds the paper's figures from the artefacts
docs/capture_paper_screenshots.py   re-captures the two dashboard figures
docs/paper_figures/                 the figures themselves
```

### The paper's figures

```bash
python -m docs.paper_figures
```

Nothing there re-trains and nothing is estimated by hand. The stored artefact
carries the fitted pipeline, the isotonic calibrators and the seed, and the data
layer is deterministic given that seed, so the test partition reproduces exactly
and every curve is the curve the evaluation reported. The two dashboard figures
need the development server running, and are re-captured at a narrow viewport —
a full-width screenshot reduced to a printed column puts its body text at about
four points:

```bash
cd bdss_project && python manage.py runserver 8811 --noreload   # in one shell
python -m docs.capture_paper_screenshots                        # in another
```

---

## The training protocol

Every model runs the same seven stages (`ml/train_common.py`), so the comparison measures the
models and not the diligence with which each was tuned:

1. **Prepare** — deduplicate, median-impute continuous fields, give categorical fields an
   explicit `Missing` level, one-hot encode, min-max scale, drop near-zero-variance predictors
   and one of any pair correlated above 0.9, split 80/20 stratified.
2. **Tune** — stratified 10-fold CV on the training partition. Grid search for logistic
   regression and the random forest; Bayesian optimisation (Optuna, TPE) for XGBoost, whose
   parameter space is too large for exhaustive search. **SMOTE sits inside the pipeline**, so
   it is applied within each fold — never across the split — and the test partition keeps its
   natural class distribution throughout.
3. **Calibrate** — isotonic regression fitted on out-of-fold probabilities. The decision layer
   consumes probabilities, and a miscalibrated 0.8 is operationally misleading. Isotonic is
   monotone, so it leaves the Shapley ranking untouched.
4. **Choose the operating threshold** — on those same validation folds, maximising F1 subject
   to a precision floor. Not 0.5. Precision decides how much of the retention budget is spent
   on customers who were never going to leave, which makes the threshold a business decision
   wearing a technical disguise.
5. **Refit and evaluate once** — a single pass over a test partition untouched by training,
   tuning, calibration or threshold selection.
6. **Explain and audit** — SHAP for every scored customer; SHAP's top-5 drivers compared
   against LIME's on 100 records stratified across the risk spectrum. Features where the two
   explainers systematically disagree are recorded, and any recommendation resting on one of
   them carries a visible caveat in the dashboard.
7. **Register** — the artefact is stored with its data snapshot (SHA-256), hyperparameters,
   evaluation report, explanation audit and fairness audit. Every score the dashboard serves
   records the model version that produced it.

**Runtime.** The random forest grid dominates: 16 candidates × 10 folds = 160 forest fits.
On a 20k-row banking sample the full ladder takes roughly 15–30 minutes. `--fast` runs the
whole thing in about two minutes with a reduced search space; `--max-rows N` subsamples.
The full 165k banking file will take hours — that is the honest cost of a 10-fold grid search,
not a bug.

### Evaluation

Accuracy alone is uninformative under imbalance — a classifier that never predicts churn is
79% accurate on the banking data — so the protocol reports precision, recall, F1, ROC-AUC,
**MCC** (the more conservative summary of a confusion matrix), and the **Brier score** for
calibration, plus a profit-oriented reading that values the confusion matrix at the operating
threshold.

The unit economics in `ml/config.py` (100 for a customer retained, 15 for an offer wasted, 200
for a customer lost) are **placeholders**. They are the first thing an adopting firm must
replace, because they — not the model — decide where the threshold lands.

Models are compared on **two axes**: discrimination and calibration on one, explanation
stability and fidelity on the other. `ml/train_all.py` prints both and recommends a model under
a deliberately conservative rule: a model whose explanations failed the audit is disqualified,
and an ensemble that beats the transparent baseline by less than one point of MCC loses to it.
An unexplainable model has to earn its opacity.

---

## Data

| Dataset | Rows | Label | Positive rate | Used |
|---|---|---|---|---|
| `datasets/Bank/train.csv` | 165,034 | `Exited` | 21.2% | **Yes** — retail banking |
| `datasets/netflix/netflix_customer_churn.csv` | 5,000 | `churned` | 50.3% | **Yes** — subscription streaming |
| `datasets/Eshop/ecommerce_customer_features.csv` | 6,000 | *none* | — | **No** — see below |
| `datasets/Bank/test.csv` | 110,023 | *none* | — | Unlabelled; a realistic upload for the dashboard |

**The e-commerce file carries no churn label**, so it supports no supervised model and is
deliberately not registered. The obvious fix — defining churn as "no purchase in N days" — is
a trap: `engagement_score` correlates at **r = −0.85** with `days_since_last_purchase`, so a
model keeping both would post a spectacular ~0.99 AUC that is pure leakage; and with both
removed, the remaining twelve predictors correlate with the label at |r| ≤ 0.024, giving a
coin-flip. Fabricating a third domain is worse than reporting two, so the cross-domain claim
here rests on banking and streaming. A labelled e-commerce file drops into `ml/config.py` as a
new `DatasetConfig` whenever one exists.

### Read the streaming metrics with suspicion

The streaming models score ROC-AUC 0.9999 and MCC 0.984. **That is not a result; it is a
warning.** Churn does not behave that way in the wild. The label in that file is synthetic: a
depth-5 decision tree recovers 93% of it from three columns, and the tree it learns is a clean
threshold rule over `avg_watch_time_per_day`, `last_login_days` and `watch_hours`. The models
are rediscovering the generator that wrote the file, not predicting a cancellation.

The dataset is still useful — it exercises every layer of the framework on a second domain with
a different feature vocabulary, which is what the cross-domain design needs it for — but the
**banking figures are the ones to judge the framework by**, because that data was not generated
by a rule. This caveat is repeated on the streaming model cards in the dashboard, where someone
might otherwise read 0.99 and believe it.

`Surname` is dropped from the banking data rather than merely unused: it is a direct identifier
and a proxy for ethnicity and nationality, and no model in a retention system has any business
reading one.

---

## Ethics, as implemented

The paper treats ethical safeguards as design requirements rather than an appendix. So does the
code.

**Fairness is audited, and the audit has teeth.** `ml/fairness.py` compares selection rates and
true-positive rates across every declared sensitive attribute, applies the four-fifths
convention, and flags demographic attributes that appear among the model's leading drivers. On
the banking data it finds real disparities and says so on the model card — this is a property
of the data, and hiding it would be the failure.

**Demographic drivers are explained but never acted on.** A customer's age or nationality may
legitimately appear among the drivers of their risk — suppressing it would make the explanation
a lie. But the decision layer matches interventions only against *actionable* drivers. If a
customer's risk is dominated by a demographic attribute, they are routed to a human, not to a
differential offer. This invariant is enforced in `ml/decision.py` and tested in
`dashboard/tests.py::test_no_offer_is_ever_made_on_a_demographic_driver`.

**The system is advisory.** Any recommendation can be overridden; an override requires a reason;
the reason, the person and the timestamp are logged. That log feeds the next model review and is
the only way to detect automation bias — the tendency to defer to the system precisely because
it is a system. **An override rate of zero is not a triumph.** The overview screen shows it for
that reason.

**Data protection.** `BDSS_LOG_INPUT_DATA=0` keeps the score, the drivers and the recommendation
while discarding the personal data behind them. The dashboard loads no external asset — charts
are rendered server-side and embedded — so a page displaying customer risk phones nowhere.
Predictions are read-only in the admin: evidence that can be edited after the fact is not
evidence.

---

## Limitations

Stated plainly, as in the paper.

- **Two domains, not three.** The e-commerce arm is unlabelled (above). Cross-domain transfer
  is evidenced across banking and streaming only.
- **Shapley attributions describe the model, not the world.** The actions the taxonomy triggers
  are motivated correlations, not established causes. The framework flags customers and reasons;
  it cannot promise that treating the reason will avert the churn.
- **The taxonomy is untested.** It encodes a judgement that a given driver is best answered by a
  given offer, resting on the retention-marketing literature and face validity, not on
  experimental evidence. It is editable precisely for that reason — but editability transfers
  the burden to the adopting firm rather than discharging it. The end-to-end value of the system
  remains a hypothesis until it is measured against realised retention outcomes, ideally against
  a holdout control group.
- **The explanation audit is evidence of robustness, not proof of fidelity.** Agreement between
  two post-hoc methods means they agree, not that either is right.
- **The dashboard is designed against personas, not validated with users.** A study with
  practising retention staff would test whether the explanations actually change decisions.
- **The streaming data is close to balanced** (50.3% churn), which is unusual for churn and makes
  SMOTE close to a no-op there. It is a useful counterweight to the banking data precisely
  because it is atypical.

The natural next step is uplift modelling, which estimates the *effect* of an intervention on
each customer and would convert the decision layer from risk-based to effect-based targeting.

---

## Testing

```bash
cd bdss_project
python manage.py test dashboard
```

Fourteen tests covering the things that would quietly break a retention decision without ever
raising an exception: that no screen showing customer risk is reachable without signing in, that
a served prediction carries the version of the model that made it, that an override is attributed
to the person who made it, that the API returns the reasons and not merely the score, and that no
offer is ever made on the basis of a demographic driver. Tests requiring a trained model skip
cleanly on a fresh checkout and run for real once `ml.train_all` has been run.

---

## Configuration

Set via environment variables; defaults are development-appropriate.

| Variable | Default | What it does |
|---|---|---|
| `DJANGO_SECRET_KEY` | dev key | **Must** be set in production |
| `DJANGO_DEBUG` | `1` | Set `0` in production; enables HSTS, secure cookies, SSL redirect |
| `BDSS_MODELS_DIR` | `models/` | Where the registry lives |
| `BDSS_RULES_PATH` | `decision/intervention_rules.json` | The editable taxonomy |
| `BDSS_MAX_UPLOAD_ROWS` | `20000` | Upload cap |
| `BDSS_TOP_DRIVERS` | `3` | Drivers surfaced per customer |
| `BDSS_LOG_INPUT_DATA` | `1` | Set `0` to retain reasons but not personal data |
