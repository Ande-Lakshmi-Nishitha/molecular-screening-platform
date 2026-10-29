# Deploy: Compose stack, health checks, backup and restore (MSP-11)

## Start
    cp .env.example .env        # then fill in POSTGRES_USER / PASSWORD / DB
    mkdir -p models             # bind-mounted read-only into worker-ml and worker-dl
    docker compose up --build

Open https://localhost (self-signed certificate; accept the browser warning).
Port 80 redirects to HTTPS. Only the proxy publishes ports.
`certs-init` runs once to create the certificate and then exits; that is expected.

## Health
- `GET /healthz`: process is up (always 200).
- `GET /readyz`: 200 if PostgreSQL and Redis answer, otherwise 503.
Both drive the Compose healthchecks. Check with `docker compose ps`.

## Backup and restore
`db-backup` writes a `pg_dump` to the `backups` volume at start and then every
`BACKUP_INTERVAL_SECONDS` (default 86400), keeping the newest `BACKUP_KEEP` (7).

    docker compose exec db-backup ls /backups
    docker compose exec db-backup /scripts/restore.sh            # newest dump
    docker compose exec db-backup /scripts/restore.sh /backups/msp_<stamp>.dump

Restore works against an empty database (`--clean --if-exists`).

## Workers
Each worker runs one queue (`ml`, `dl`, `qml`, `agg`) with prefetch 1.
`QML_CONCURRENCY` sets the QML worker's concurrency (1 or 2, NFR-013).
Until `backend/celery_app.py` exists (MSP-29) workers run as idle placeholders.
