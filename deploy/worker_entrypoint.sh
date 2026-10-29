#!/bin/sh
# Starts a Celery worker for one queue. If backend/celery_app.py does not
# define a Celery app yet (MSP-29), it idles as a placeholder instead.
set -eu
: "${QUEUE:?QUEUE must be set (ml, dl, qml or agg)}"
CONCURRENCY="${CONCURRENCY:-1}"

HAS_APP='import sys
try:
    from backend import celery_app as m
except Exception:
    sys.exit(1)
sys.exit(0 if (hasattr(m, "app") or hasattr(m, "celery")) else 1)'

if python -c "$HAS_APP"; then
    exec celery -A backend.celery_app worker -Q "$QUEUE" -c "$CONCURRENCY" \
        --prefetch-multiplier=1 -Ofair --hostname "${QUEUE}@%h"
fi

echo "worker-${QUEUE}: no Celery app in backend.celery_app yet, running as placeholder"
while true; do
    touch /tmp/worker.alive
    sleep 20
done