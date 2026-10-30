#!/bin/sh
# Starts a Celery worker for one queue. It imports the queue's task module
# (backend.workers.<queue>.tasks) so the task registers; without that import the
# worker starts but cannot run msp.<queue>.run.
# If backend/celery_app.py does not define a Celery app, it idles as a placeholder.
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
        -I "backend.workers.${QUEUE}.tasks" \
        --prefetch-multiplier=1 -Ofair --hostname "${QUEUE}@%h"
fi

echo "worker-${QUEUE}: no Celery app in backend.celery_app yet, running as placeholder"
while true; do
    touch /tmp/worker.alive
    sleep 20
done
