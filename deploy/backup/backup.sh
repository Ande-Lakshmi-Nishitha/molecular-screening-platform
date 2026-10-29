#!/bin/sh
# Scheduled pg_dump (QA-REC). Runs once immediately, then every
# BACKUP_INTERVAL_SECONDS. Keeps the newest BACKUP_KEEP dumps.
set -eu
BACKUP_DIR="${BACKUP_DIR:-/backups}"
INTERVAL="${BACKUP_INTERVAL_SECONDS:-86400}"
KEEP="${BACKUP_KEEP:-7}"

while true; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    out="${BACKUP_DIR}/msp_${stamp}.dump"
    if pg_dump -h postgres -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f "${out}.tmp"; then
        mv "${out}.tmp" "$out"
        touch /tmp/last_backup_ok
        echo "backup written: $out"
        ls -1t "${BACKUP_DIR}"/msp_*.dump | tail -n +$((KEEP + 1)) | xargs -r rm -f
    else
        rm -f "${out}.tmp"
        echo "backup FAILED" >&2
    fi
    sleep "$INTERVAL"
done
