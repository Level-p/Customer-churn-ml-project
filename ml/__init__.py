"""Machine learning pipeline for the Business Decision Support System (BDSS).

The package implements the five-layer architecture described in the underlying
research (Onoja, "Predicting Customer Churn in Subscription Services Using
Interpretable Machine Learning"), retargeted at sales / demand forecasting:

    data layer        -> ml.data_prep
    feature layer     -> ml.features
    model layer       -> ml.train_logistic_regression / _random_forest / _xgboost
    explanation layer -> ml.explain          (SHAP, audited against LIME)
    decision layer    -> ml.decision         (intervention taxonomy)

Trained artefacts are written to the versioned registry in ``ml.registry`` and
consumed at inference time by the Django application in ``bdss_project``.
"""

__version__ = "1.0.0"
