#!/bin/sh
# Healthy if the Celery worker answers ping, or (placeholder mode) the
# heartbeat file is fresh.
set -eu
HAS_APP='import sys
try:
    from backend import celery_app as m
except Exception:
    sys.exit(1)
sys.exit(0 if (hasattr(m, "app") or hasattr(m, "celery")) else 1)'

if python -c "$HAS_APP"; then
    celery -A backend.celery_app inspect ping -d "${QUEUE}@$(hostname)" -t 5 >/dev/null
else
    test -n "$(find /tmp/worker.alive -mmin -1 2>/dev/null)"
fi