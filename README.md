# Molecule Screening Platform (MSP)

Web app for preliminary computational screening of molecules with ML, DL and QML tiers. Predictions are indicative estimates, not experimental results, and nothing here says a molecule is safe or effective.

A molecule goes in as a SMILES string (typed, or picked from a PubChem or ChEMBL search). It is checked and standardised with RDKit, and each tier that applies to it runs as a background job on its own queue:

| Tier | What it gives | Worker |
| --- | --- | --- |
| ML | Aqueous solubility (LogS) with a 90% interval, and 12 Tox21 assay signals | `worker-ml` |
| DL | Target activity (pIC50 or pKi) from a 5-model GNN ensemble, with a baseline | `worker-dl` |
| QML | Electronic excitation gap for small molecules (the simulated quantum backend is labelled as a stub) | `worker-qml` |

A tier that does not apply (for example QML for a molecule with more than nine heavy atoms, or DL when no target is chosen) is shown as skipped, with the reason. A failed tier never hides the results of the others.

## Setup

### What you need

- **Docker Desktop** (Docker Compose v2) and **Git**.
- **Node.js 24** to run the front end (`node -v`). Older versions fail on the front end's dependencies.
- **Python 3.11 or newer** only if you want to run the Python tests outside Docker.
- About 5 GB of free disk space. The first build downloads RDKit, XGBoost and the CPU build of PyTorch and takes several minutes.

### 1. Get the code and configure it

```
git clone <repository-url>
cd molecular-screening-platform
```

Copy the settings file (`cp .env.example .env` on macOS and Linux, `copy .env.example .env` on Windows) and open `.env`. Four values are required:

| Name | Value |
| --- | --- |
| `POSTGRES_USER` | any user name, for example `msp` |
| `POSTGRES_PASSWORD` | any password |
| `POSTGRES_DB` | any database name, for example `msp` |
| `JWT_SECRET` | a long random string. Make one with `python -c "import secrets; print(secrets.token_hex(32))"` |

The other names in `.env.example` are optional overrides with defaults, and the Compose stack does not read them. `.env` holds secrets and is never committed.

### 2. Add the model files

The trained models are not in Git. Put the exported files in `models/ml` and `models/dl` (get them from the team, or produce them with the scripts in `training/`). Without them the workers start but every ML and DL task fails with `MODEL_UNAVAILABLE`.

### 3. Start the stack

```
docker compose up --build -d
docker compose ps
```

Wait about a minute until every service shows `healthy`. `certs-init` exits on purpose after creating the certificate. Only the proxy publishes ports (80 and 443).

### 4. Create the database tables

```
docker compose exec api alembic upgrade head
```

Run this once on a new database. It does nothing if the tables are already current. If you ever remove the volumes (`docker compose down -v`), run it again.

### 5. Check it works

```
curl -k https://localhost/readyz
```

You should see `"status":"ready"`. The `-k` is needed because the certificate is self-signed.

### 6. Start the front end

```
cd frontend
npm install
npm run dev
```

Open the address it prints (usually `http://localhost:5173`). It forwards `/api` to the stack at `https://localhost`.

## Usage

1. **Register**, then **log in**. After a period of inactivity the session expires and you are sent back to the login page.
2. **Submit** a molecule: type a SMILES string (for example `CCO` for ethanol), or use **Search** to find one in PubChem or ChEMBL and send it to the submit page. Choose a target if you want a DL prediction. Invalid input is rejected with a clear message.
3. **Track** the job. Each tier shows its own state (queued, running, done, failed, skipped, cancelled), and a finished tier shows its result while the others are still running. **Cancel** is offered while the job is queued or running; finished results are kept.
4. Read the **results**: the triage summary, each tier's prediction with units, warnings and the limitation statement for that result. **History** lists your own jobs, and a job's report can be exported as JSON or PDF.
5. **Model info** shows each model's version, training data and published validation metrics.

You only ever see your own jobs.

### Using the API directly

The OpenAPI description is served at `https://localhost/api/openapi.json`. Every endpoint except register and login needs `Authorization: Bearer <token>`.

```
curl -k -X POST https://localhost/api/auth/login -H "Content-Type: application/json" -d "{\"email\":\"you@example.com\",\"password\":\"your password\"}"
curl -k -X POST https://localhost/api/molecules/submit -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d "{\"smiles\":\"CCO\"}"
curl -k https://localhost/api/jobs/<job_id> -H "Authorization: Bearer <token>"
```

Errors always have the same shape: `{"error": {"code", "message", "details", "request_id"}}`. The same `request_id` is in the `X-Request-ID` response header and in the server log.

## Tests

```
python -m unittest discover -s tests -t .       # Python: unit, integration, failure injection
cd frontend && npm test && npm run lint          # front end
```

Some tests skip when a service they need is not running. A skipped test is not a passed test.

## Backup and restore

The database is backed up automatically. See [deploy/README.md](deploy/README.md) for the schedule and the restore command.

## Troubleshooting

| What you see | Cause and fix |
| --- | --- |
| `docker compose` stops with `set JWT_SECRET in .env` | `JWT_SECRET` is missing from `.env`. Add it. |
| Login or register returns `internal_error` (500) | The message is deliberately generic. Look at `docker compose logs api --tail 40`. |
| The log says `relation "users" does not exist` | The tables were never created. Run `docker compose exec api alembic upgrade head`. |
| The log says `JWT_SECRET is not set` | Add it to `.env`, then run `docker compose up -d`. |
| A worker logs `ModelLoadError`, or tasks fail with `MODEL_UNAVAILABLE` | `models/ml` or `models/dl` is empty or incomplete. Add the model files and run `docker compose restart worker-ml worker-dl`. |
| `worker-dl` keeps restarting with `No module named 'torch'` | The image was built before PyTorch was added. Run `docker compose build worker-dl`. |
| A service stays `unhealthy` | Run `docker compose logs <service> --tail 30`. |
| Port 80 or 443 is already in use | Stop the other web server, or change the proxy ports in `docker-compose.yml`. |
| The browser warns about the certificate | It is self-signed on purpose. Accept it, or use `curl -k`. |
| `npm run dev` fails at install or start | Check `node -v`. Use Node 24. |
| A `.sh` file fails inside a container with `bad interpreter` or `not found` | It was checked out with Windows line endings. The repo's `.gitattributes` forces LF for `*.sh`. Re-clone, or run `git add --renormalize .` and rebuild. |
| The front end shows a 401 or sends you to login | The session expired or the token is invalid. Log in again. |
| Search returns an error naming PubChem or ChEMBL | That service is down or slow. Type the SMILES string directly instead. |

## What is not built

Differences between the finished system and the SRS, with reasons, are listed in [docs/deviation-log.md](docs/deviation-log.md).

## Team workflow

See CONTRIBUTING.md.
