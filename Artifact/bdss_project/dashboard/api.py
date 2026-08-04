"""REST API.

    GET  /api/models/    the models the registry can currently serve
    POST /api/predict/   score customers and return the recommended action

The API returns exactly what the dashboard shows -- probability, drivers,
recommendation, owner, priority, model version -- because a CRM that consumes the
score without the reasons reintroduces the precise problem the framework was
built to solve. Every call is authenticated, and every prediction it makes is
written to the same audit trail as a prediction made through the interface.
"""

from __future__ import annotations

import logging

import pandas as pd
from django.conf import settings
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from dashboard.models import Prediction
from dashboard.serializers import PredictRequestSerializer, PredictionResultSerializer
from ml_engine import predictor

LOGGER = logging.getLogger(__name__)


class ModelListView(APIView):
    """The registry, as served: what can be called, and how well it scores."""

    def get(self, request: Request) -> Response:
        models = predictor.available_models()
        return Response(
            {
                "count": len(models),
                "models": models,
                "default": _default_model_id(models),
                "usage": (
                    "POST /api/predict/ with {'model': '<id>', 'records': [{...}]}. "
                    "Omit 'model' to use the default."
                ),
            }
        )


class PredictView(APIView):
    """Score one or more customers and return an explained recommendation."""

    def post(self, request: Request) -> Response:
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        records = payload["records"]
        models = predictor.available_models()
        if not models:
            return Response(
                {
                    "detail": "No models are registered. Train one with "
                    "`python -m ml.train_all`."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        model_id = payload.get("model") or _default_model_id(models)
        frame = pd.DataFrame(records)

        try:
            results, meta = predictor.predict(
                frame,
                identifier=model_id,
                top_drivers=payload.get("top_drivers", settings.BDSS_TOP_DRIVERS),
                reference_column=payload.get("reference_column"),
            )
        except predictor.ModelUnavailable as exc:
            return Response(
                {"detail": str(exc), "available": [model["id"] for model in models]},
                status=status.HTTP_404_NOT_FOUND,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("API scoring failed for model %s", model_id)
            return Response(
                {
                    "detail": (
                        "The records could not be scored with this model, most likely "
                        f"because they do not match its feature schema. ({exc})"
                    ),
                    "expected_columns_missing": [],
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        _log_predictions(request, results, meta)

        body = {
            "model": {
                "id": meta["model_id"],
                "label": meta["model_label"],
                "version": meta["model_version"],
                "dataset": meta["dataset"],
                "threshold": meta["threshold"],
                "trained_at": meta["trained_at"],
            },
            "taxonomy_version": meta["taxonomy_version"],
            "count": len(results),
            # An honest response says what it did not receive. A silently imputed
            # column is the difference between a score and a guess.
            "missing_columns": meta["missing_columns"],
            "ignored_columns": meta["ignored_columns"],
            "predictions": PredictionResultSerializer(results, many=True).data,
            "notice": (
                "Shapley attributions describe the behaviour of the model, not the "
                "causes of the customer's behaviour. Recommendations are advisory and "
                "may be overridden."
            ),
        }
        return Response(body, status=status.HTTP_200_OK)


def _default_model_id(models: list[dict]) -> str:
    """The best registered model by F1, so a caller can omit the choice."""
    best = max(models, key=lambda model: model["metrics"].get("f1") or 0)
    return best["id"]


def _log_predictions(request: Request, results: list[dict], meta: dict) -> None:
    """Write API predictions to the same audit trail as the dashboard's.

    A prediction made over HTTP is still a prediction about a real customer, and
    it earns exactly the same accountability as one made through the interface.
    """
    keep_inputs = settings.BDSS_LOG_INPUT_DATA
    Prediction.objects.bulk_create(
        [
            Prediction(
                batch=None,
                user=request.user if request.user.is_authenticated else None,
                source="api",
                customer_ref=str(result["customer_ref"])[:128],
                input_data=result["input_data"] if keep_inputs else {},
                model_key=meta["model_key"],
                model_label=meta["model_label"],
                model_version=meta["model_version"],
                dataset=meta["dataset"],
                threshold=meta["threshold"],
                predicted_label=str(result["predicted_label"]),
                predicted_churn=result["predicted_churn"],
                probability=result["probability"],
                confidence=result["confidence"],
                band=result["band"],
                top_drivers=result["drivers"],
                recommendation=result["recommendation"]["intervention"][:255],
                business_owner=result["recommendation"]["owner"][:64],
                priority=result["priority"],
                priority_score=result["priority_score"],
                rule_id=result["recommendation"]["rule_id"][:64],
                rationale=result["recommendation"]["rationale"],
                taxonomy_version=str(meta.get("taxonomy_version") or ""),
                caveats=result["recommendation"]["caveats"],
            )
            for result in results
        ],
        batch_size=500,
    )
