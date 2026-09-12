#!/usr/bin/env bash
# Container entrypoint: apply any pending migrations against the configured
# database (SQLite in the single-container image), then hand control to
# supervisord, which keeps redis + celery + daphne alive.
set -euo pipefail

cd /app/backend

echo "[entrypoint] applying migrations..."
python manage.py migrate --noinput

echo "[entrypoint] starting supervisord (redis + celery + daphne)..."
exec supervisord -c /etc/supervisord.conf