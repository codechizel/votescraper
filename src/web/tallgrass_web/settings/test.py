"""Test settings — fast password hashing, dedicated test database."""

from .base import *  # noqa: F401, F403

DEBUG = False

# Fast password hashing for test speed.
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

DATABASES["default"]["TEST"] = {"NAME": "tallgrass_test"}  # noqa: F405
