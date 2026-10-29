#!/bin/sh
# Restore a dump into the database. Usage: restore.sh [dump-file]
# With no argument the newest dump in BACKUP_DIR is used.
set -eu
BACKUP_DIR="${BACKUP_DIR:-/backups}"
file="${1:-$(ls -1t "${BACKUP_DIR}"/msp_*.dump 2>/dev/null | head -n 1)}"
if [ -z "$file" ] || [ ! -f "$file" ]; then
    echo "no backup file found" >&2
    exit 1
fi
echo "restoring $file"
pg_restore -h postgres -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    --clean --if-exists --no-owner "$file"
echo "restore finished"
