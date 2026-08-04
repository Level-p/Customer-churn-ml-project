"""Tests for the decision-support layer.

These are not unit tests of scikit-learn. They test the things that would quietly
break a retention decision and never raise an exception:

* a page that shows customer risk must not be reachable without signing in;
* a served prediction must carry the version of the model that made it;
* an override must be attributed to the person who made it;
* the API must return the reasons, not merely the score.

The tests that need a trained model are skipped when the registry is empty, so
the suite runs on a clean checkout, and runs for real once the models are trained.
"""

from __future__ import annotations

import io
import json
import unittest

import pandas as pd
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from dashboard.models import Prediction, PredictionBatch
from ml_engine import predictor

MODELS = predictor.available_models()
requires_model = unittest.skipIf(
    not MODELS, "no trained model in the registry; run `python -m ml.train_all` first"
)


def _sample_frame(dataset: str) -> pd.DataFrame:
    """A couple of rows in the shape the given model was trained on."""
    if dataset == "bank_churn":
        return pd.DataFrame(
            [
                {
                    "CustomerId": 15634602, "CreditScore": 619, "Geography": "France",
                    "Gender": "Female", "Age": 42, "Tenure": 2, "Balance": 0.0,
                    "NumOfProducts": 1, "HasCrCard": 1.0, "IsActiveMember": 1.0,
                    "EstimatedSalary": 101348.88,
                },
                {
                    "CustomerId": 15647311, "CreditScore": 608, "Geography": "Germany",
                    "Gender": "Male", "Age": 58, "Tenure": 1, "Balance": 125510.82,
                    "NumOfProducts": 1, "HasCrCard": 0.0, "IsActiveMember": 0.0,
                    "EstimatedSalary": 112542.58,
                },
            ]
        )
    return pd.DataFrame(
        [
            {
                "customer_id": "a-1", "age": 51, "gender": "Other",
                "subscription_type": "Basic", "watch_hours": 14.7,
                "last_login_days": 29, "region": "Africa", "device": "TV",
                "monthly_fee": 8.99, "payment_method": "Gift Card",
                "number_of_profiles": 1, "avg_watch_time_per_day": 0.49,
                "favorite_genre": "Action",
            }
        ]
    )


class AccessControlTests(TestCase):
    """Customer risk is not public, and the system does not forget who asked."""

    def test_every_screen_requires_a_signed_in_user(self):
        for name in ("dashboard:index", "dashboard:upload", "dashboard:history",
                     "dashboard:taxonomy", "dashboard:registry"):
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 302, name)
            self.assertIn("/accounts/login/", response["Location"], name)

    def test_api_rejects_anonymous_callers(self):
        response = self.client.post(
            reverse("api-predict"),
            data=json.dumps({"records": [{"Age": 40}]}),
            content_type="application/json",
        )
        self.assertIn(response.status_code, (401, 403))


class TaxonomyTests(TestCase):
    """The rule table is published, because a rule nobody can read is a rule
    nobody can challenge."""

    def setUp(self):
        self.user = get_user_model().objects.create_user("planner", password="pw-for-tests-1")
        self.client = Client()
        self.client.force_login(self.user)

    def test_taxonomy_is_visible_to_a_signed_in_user(self):
        response = self.client.get(reverse("dashboard:taxonomy"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Intervention taxonomy")


@requires_model
class ScoringTests(TestCase):
    """The scoring path, end to end, with a real registered model."""

    def setUp(self):
        self.user = get_user_model().objects.create_user("analyst", password="pw-for-tests-2")
        self.client = Client()
        self.client.force_login(self.user)
        self.model = MODELS[0]
        self.frame = _sample_frame(self.model["dataset"])

    def test_predictor_returns_drivers_and_a_recommendation(self):
        results, meta = predictor.predict(self.frame, identifier=self.model["id"])

        self.assertEqual(len(results), len(self.frame))
        self.assertEqual(meta["model_version"], self.model["version"])

        for result in results:
            self.assertTrue(0.0 <= result["probability"] <= 1.0)
            self.assertTrue(result["drivers"], "a score with no reasons is the failure mode")
            self.assertTrue(result["recommendation"]["intervention"])
            self.assertTrue(result["recommendation"]["owner"])
            self.assertIn(result["priority"], {"P1", "P2", "P3"})

    def test_no_offer_is_ever_made_on_a_demographic_driver(self):
        """The strongest driver may be demographic; the recommendation may not rest on it."""
        results, _ = predictor.predict(self.frame, identifier=self.model["id"])
        for result in results:
            recommendation = result["recommendation"]
            triggering = recommendation.get("triggering_feature")
            if triggering is None:
                continue
            driver = next(
                (d for d in result["drivers"] if d["feature"] == triggering), None
            )
            if driver and driver["protected"]:
                # The only rule permitted to fire on a protected driver is the one
                # that refuses to make an offer at all.
                self.assertEqual(recommendation["rule_id"], "demographic_no_offer")

    def test_upload_creates_a_batch_and_logs_every_prediction(self):
        csv_bytes = self.frame.to_csv(index=False).encode("utf-8")
        upload = SimpleUploadedFile("customers.csv", csv_bytes, content_type="text/csv")

        response = self.client.post(
            reverse("dashboard:upload"),
            {"csv_file": upload, "model_id": self.model["id"]},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        batch = PredictionBatch.objects.latest("created_at")
        self.assertEqual(batch.row_count, len(self.frame))
        self.assertEqual(batch.model_version, self.model["version"])
        self.assertEqual(batch.predictions.count(), len(self.frame))

        # Provenance: a score that cannot name the model that made it is not evidence.
        for prediction in batch.predictions.all():
            self.assertEqual(prediction.model_version, self.model["version"])
            self.assertTrue(prediction.top_drivers)
            self.assertTrue(prediction.recommendation)

    def test_an_override_is_recorded_against_the_person_who_made_it(self):
        csv_bytes = self.frame.to_csv(index=False).encode("utf-8")
        self.client.post(
            reverse("dashboard:upload"),
            {
                "csv_file": SimpleUploadedFile("c.csv", csv_bytes, content_type="text/csv"),
                "model_id": self.model["id"],
            },
        )
        prediction = Prediction.objects.first()

        self.client.post(
            reverse("dashboard:override", args=[prediction.pk]),
            {"action": "already_resolved", "reason": "Customer renewed last week."},
        )

        prediction.refresh_from_db()
        self.assertTrue(prediction.is_overridden)
        self.assertEqual(prediction.overridden_by, self.user)
        self.assertEqual(prediction.override_reason, "Customer renewed last week.")
        self.assertIsNotNone(prediction.overridden_at)

    def test_override_without_a_reason_is_refused(self):
        csv_bytes = self.frame.to_csv(index=False).encode("utf-8")
        self.client.post(
            reverse("dashboard:upload"),
            {
                "csv_file": SimpleUploadedFile("c.csv", csv_bytes, content_type="text/csv"),
                "model_id": self.model["id"],
            },
        )
        prediction = Prediction.objects.first()

        self.client.post(
            reverse("dashboard:override", args=[prediction.pk]), {"action": "no_action"}
        )
        prediction.refresh_from_db()
        self.assertFalse(prediction.is_overridden)

    def test_export_carries_the_reasons_and_the_model_version(self):
        csv_bytes = self.frame.to_csv(index=False).encode("utf-8")
        self.client.post(
            reverse("dashboard:upload"),
            {
                "csv_file": SimpleUploadedFile("c.csv", csv_bytes, content_type="text/csv"),
                "model_id": self.model["id"],
            },
        )
        batch = PredictionBatch.objects.latest("created_at")

        response = self.client.get(reverse("dashboard:export", args=[batch.pk]))
        self.assertEqual(response.status_code, 200)

        text = response.content.decode("utf-8")
        self.assertIn("recommended_intervention", text)
        self.assertIn("driver_1", text)
        self.assertIn(batch.model_version, text)

    def test_a_junk_upload_fails_with_an_explanation_not_a_stack_trace(self):
        junk = SimpleUploadedFile("junk.csv", b"", content_type="text/csv")
        response = self.client.post(
            reverse("dashboard:upload"), {"csv_file": junk, "model_id": self.model["id"]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "empty")


@requires_model
class ApiTests(TestCase):
    """The API returns the reasons, not merely the score."""

    def setUp(self):
        self.user = get_user_model().objects.create_user("service", password="pw-for-tests-3")
        self.client = Client()
        self.client.force_login(self.user)
        self.model = MODELS[0]

    def test_models_endpoint_lists_the_registry(self):
        response = self.client.get(reverse("api-models"))
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json()["count"], 1)

    def test_predict_returns_probability_drivers_and_recommendation(self):
        record = _sample_frame(self.model["dataset"]).iloc[0].to_dict()
        response = self.client.post(
            reverse("api-predict"),
            data=json.dumps({"model": self.model["id"], "records": [record]}, default=str),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.content[:400])

        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["model"]["version"], self.model["version"])

        prediction = body["predictions"][0]
        self.assertIn("probability", prediction)
        self.assertIn("confidence", prediction)
        self.assertTrue(prediction["drivers"])
        self.assertTrue(prediction["recommendation"]["intervention"])
        self.assertTrue(prediction["recommendation"]["owner"])

    def test_api_predictions_are_written_to_the_same_audit_trail(self):
        record = _sample_frame(self.model["dataset"]).iloc[0].to_dict()
        self.client.post(
            reverse("api-predict"),
            data=json.dumps({"model": self.model["id"], "records": [record]}, default=str),
            content_type="application/json",
        )
        logged = Prediction.objects.filter(source="api")
        self.assertEqual(logged.count(), 1)
        self.assertEqual(logged.first().model_version, self.model["version"])

    def test_an_unknown_model_is_refused_rather_than_substituted(self):
        response = self.client.post(
            reverse("api-predict"),
            data=json.dumps({"model": "gradient_boosted_unicorn|bank_churn", "records": [{"Age": 40}]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 404)
