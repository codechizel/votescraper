"""Shared Django settings for Tallgrass.

Database, installed apps, middleware, and project-wide defaults.
Environment-specific overrides live in local.py, test.py, etc.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-only-change-in-production",
)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "legislature",
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

ROOT_URLCONF = "tallgrass_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "tallgrass_web.wsgi.application"

# -- Database ----------------------------------------------------------------
# Default: local PostgreSQL matching docker-compose.yml.
# Override with DATABASE_URL env var in production.

_db_url = os.environ.get("DATABASE_URL", "postgres://tallgrass:tallgrass@localhost:5432/tallgrass")


def _parse_database_url(url: str) -> dict:
    """Parse a postgres:// URL into Django DATABASES dict."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": parsed.path.lstrip("/"),
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": parsed.hostname or "localhost",
        "PORT": str(parsed.port or 5432),
    }


DATABASES = {
    "default": _parse_database_url(_db_url),
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# -- Internationalization ----------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = "America/Chicago"
USE_I18N = False
USE_TZ = True

# -- Static files ------------------------------------------------------------

STATIC_URL = "static/"
