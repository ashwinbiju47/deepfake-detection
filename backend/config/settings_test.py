"""Test-only settings overriding the database to in-memory SQLite.

The production settings target PostgreSQL (design Technology Mapping). For the
pure persistence-layer tests (Task 2.2) no Postgres-specific behavior is
exercised, so this override lets the ``@pytest.mark.django_db`` tests run on
SQLite in environments without a running Postgres instance.

Usage:
    DJANGO_SETTINGS_MODULE=config.settings_test pytest
"""

from .settings import *  # noqa: F401,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
