"""Inference engine: loads registered models and scores customers.

``predictor`` owns the model cache and the scoring path; ``utils`` owns the
input handling, the summaries and the charts. Neither reimplements any modelling
logic: both delegate to the ``ml`` package that trained the artefacts.
"""
