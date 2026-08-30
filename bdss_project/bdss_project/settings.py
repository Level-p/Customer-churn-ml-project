"""Django settings for the Business Decision Support System.

The web tier is deliberately thin. It authenticates a user, takes a customer
list, hands it to the model registry produced by the training pipeline, and
renders what comes back. All of the modelling, explanation and decision logic
lives in the ``ml`` package at the repository root, which is imported here rather
than duplicated -- so what the dashboard shows a retention manager is, by
construction, what the training run measured and registered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# bdss_project/bdss_project/settings.py -> bdss_project/ -> repository root
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get("BDSS_REPO_ROOT", BASE_DIR.parent))

# The ``ml`` package is the single source of truth for the feature vocabulary,
# the calibration maths and the intervention taxonomy. Importing it keeps the
# served explanation identical to the trained one; re-implementing it here would
# guarantee the two drift apart.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #

# Development default. In production the key must come from the environment, and
# the process should refuse to start without one.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me-before-deployment"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")
    if host.strip()
]
CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]

if not DEBUG:
    # Customer data travels over this connection; it does not travel in clear.
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"

# --------------------------------------------------------------------------- #
# Applications
# --------------------------------------------------------------------------- #

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "bdss_project.urls"
WSGI_APPLICATION = "bdss_project.wsgi.application"
ASGI_APPLICATION = "bdss_project.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard:index"
LOGOUT_REDIRECT_URL = "login"

# Retention decisions are made during a working session, not left open overnight.
SESSION_COOKIE_AGE = 60 * 60 * 8
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# --------------------------------------------------------------------------- #
# REST framework
# --------------------------------------------------------------------------- #

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    # Nothing about a customer's churn risk is public. Every endpoint requires an
    # authenticated caller, and every call is written to the prediction log.
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.UserRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"user": os.environ.get("BDSS_API_RATE", "120/minute")},
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
}

# --------------------------------------------------------------------------- #
# Internationalisation and static files
# --------------------------------------------------------------------------- #

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "Europe/London"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "media/"

# --------------------------------------------------------------------------- #
# BDSS
# --------------------------------------------------------------------------- #

#: Where the training pipeline writes its artefacts, and where the dashboard
#: reads them from. One registry, one source of truth.
BDSS_MODELS_DIR = Path(os.environ.get("BDSS_MODELS_DIR", REPO_ROOT / "models"))

#: The editable intervention taxonomy. Changing this file changes what the
#: dashboard recommends, with no redeployment and no developer involved.
BDSS_RULES_PATH = Path(
    os.environ.get("BDSS_RULES_PATH", REPO_ROOT / "decision" / "intervention_rules.json")
)

#: Upload guards. A retention analyst uploads a customer list, not a database.
BDSS_MAX_UPLOAD_BYTES = int(os.environ.get("BDSS_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
BDSS_MAX_UPLOAD_ROWS = int(os.environ.get("BDSS_MAX_UPLOAD_ROWS", 20000))

#: Rows returned to the browser per batch. The full scored list is always
#: available through the CSV export; the table is paged so the page stays usable.
BDSS_RESULTS_PAGE_SIZE = int(os.environ.get("BDSS_RESULTS_PAGE_SIZE", 50))

#: Drivers surfaced per customer. Three is what a retention manager can hold in
#: their head while making a call.
BDSS_TOP_DRIVERS = int(os.environ.get("BDSS_TOP_DRIVERS", 3))

#: Retain the raw uploaded feature values alongside each logged prediction.
#: Turn this off where data minimisation requires that only the score and the
#: drivers are retained; the audit trail then keeps the reasons but not the data.
BDSS_LOG_INPUT_DATA = os.environ.get("BDSS_LOG_INPUT_DATA", "1") == "1"

FILE_UPLOAD_MAX_MEMORY_SIZE = BDSS_MAX_UPLOAD_BYTES
DATA_UPLOAD_MAX_MEMORY_SIZE = BDSS_MAX_UPLOAD_BYTES

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "standard"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "django.utils.autoreload": {"level": "WARNING"},
        "matplotlib": {"level": "WARNING"},
        "shap": {"level": "WARNING"},
    },
}
