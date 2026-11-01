"""MSP-41 load test: 10 users, 20 queued jobs, ML p95 < 1 s, DL p95 < 3 s.

Traces to: NFR-001, NFR-002, NFR-003, NFR-012, R08. Run by hand against the reference stack
(never in CI), after `docker compose up` and with model files in models/ml and models/dl:

    set "MSP_PASSWORD=Load-Test-1234!x"
    set "MSP_TARGET_ID=EGFR"
    python -m locust -f tests/performance/locustfile.py --headless -u 10 -r 10 -t 10m --host https://localhost

Each simulated user has its own account (loadtest1..N@example.test), because the API counts
60 requests a minute per user. Tier time is taken from the server's own timestamps (job
created_at to task finished_at), so it does not depend on how often the script polls.
The run is VALID only if there were no failed login/submit/status calls, no 429s, and both
tiers finished "done" on every job. An invalid run exits with code 1 and is written to
last_run_INVALID.txt: do not commit it. Limits and hardware are open items (TBD-C10).
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods
# pylint: disable=import-error,import-outside-toplevel,attribute-defined-outside-init
# pylint: disable=consider-using-f-string
import os
import platform
import threading
import time
from datetime import datetime

from locust import HttpUser, constant, events, task
from locust.exception import StopUser

JOBS = int(os.environ.get("MSP_JOBS", "20"))
P95_LIMIT_MS = {"ML": float(os.environ.get("MSP_ML_P95_MS", "1000")),
                "DL": float(os.environ.get("MSP_DL_P95_MS", "3000"))}
PASSWORD = os.environ.get("MSP_PASSWORD", "")
EMAIL_TEMPLATE = os.environ.get("MSP_EMAIL_TEMPLATE", "loadtest{n}@example.test")
REGISTER_IF_MISSING = os.environ.get("MSP_REGISTER", "1") == "1"
SMILES = os.environ.get("MSP_SMILES", "CCO")
TARGET_ID = os.environ.get("MSP_TARGET_ID", "")
POLL_SECONDS = float(os.environ.get("MSP_POLL_SECONDS", "1.0"))
JOB_TIMEOUT = float(os.environ.get("MSP_JOB_TIMEOUT", "180"))
RESULT_FILE = os.environ.get("MSP_RESULT_FILE", "tests/performance/last_run.txt")
FINAL = {"done", "failed", "skipped", "cancelled", "canceled"}

_LOCK = threading.Lock()
_COUNT = {"users": 0, "started": 0, "finished": 0}
_INVALID = {}          # reason -> how many times
_TIMING = {"server": 0, "client": 0}


def _invalid(reason):
    with _LOCK:
        _INVALID[reason] = _INVALID.get(reason, 0) + 1


def hardware():
    """What the numbers were measured on (the acceptance criteria require this)."""
    info = {"machine": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(), "python": platform.python_version()}
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        info["ram_gb"] = "unknown (pip install psutil to record it)"
    return info


def _parse(stamp):
    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))


@events.request.add_listener
def _watch_requests(request_type, name, exception=None, response=None, **_kw):
    if request_type == "TIER":
        return
    if response is not None and getattr(response, "status_code", 0) == 429:
        _invalid("HTTP 429 (rate limited) on " + name)
    elif exception is not None:
        _invalid("failed call: " + name)


class Researcher(HttpUser):
    wait_time = constant(0)

    def _login(self, email):
        with self.client.post("/api/auth/login", json={"email": email, "password": PASSWORD},
                              name="login", catch_response=True) as resp:
            body = {}
            try:
                body = resp.json()
            except ValueError:
                pass
            token = body.get("token") or body.get("access_token")
            if resp.ok and token:
                return token
            resp.failure("login failed")
            self._login_status = (resp.status_code, resp.text[:200])
            return None

    def on_start(self):
        self.client.verify = False  # self-signed certificate on the test stack
        with _LOCK:
            _COUNT["users"] += 1
            n = _COUNT["users"]
        email = EMAIL_TEMPLATE.format(n=n)
        if REGISTER_IF_MISSING:
            # "already registered" is normal on a second run, so no answer counts as a failure
            with self.client.post("/api/auth/register", json={"email": email, "password": PASSWORD},
                                  name="register", catch_response=True) as reg:
                reg.success()
        token = self._login(email)
        if not token:
            print("\nLOGIN FAILED for %s: %s" % (email, getattr(self, "_login_status", "?")))
            print("Use the same MSP_PASSWORD the accounts were registered with.")
            self.environment.runner.quit()  # nothing can run without a login
            raise StopUser()
        self.client.headers["Authorization"] = "Bearer " + token

    @task
    def screen_one_molecule(self):
        with _LOCK:
            if _COUNT["started"] >= JOBS:
                raise StopUser()
            _COUNT["started"] += 1
        try:
            self._run_job()
        finally:
            with _LOCK:
                _COUNT["finished"] += 1
                last = _COUNT["finished"] >= JOBS
            if last:
                self.environment.runner.quit()

    def _run_job(self):
        body = {"smiles": SMILES}
        if TARGET_ID:
            body["target_id"] = TARGET_ID
        t0 = time.perf_counter()
        with self.client.post("/api/molecules/submit", json=body, name="submit",
                              catch_response=True) as resp:
            if resp.status_code != 202:
                resp.failure("submit returned HTTP %s" % resp.status_code)
                with _LOCK:
                    if not _COUNT.get("submit_reported"):
                        _COUNT["submit_reported"] = True
                        print("\nSUBMIT REJECTED: HTTP %s, body: %s" % (resp.status_code, resp.text[:400]))
                        print("request body was: %s" % body)
                return
            job_id = resp.json()["job_id"]
        seen = set()
        while time.perf_counter() - t0 < JOB_TIMEOUT:
            reply = self.client.get("/api/jobs/" + job_id, name="/api/jobs/[id]")
            if not reply.ok:
                time.sleep(POLL_SECONDS)
                continue
            job = reply.json()
            for item in job["tasks"]:
                tier = item["tier"].upper()
                if item["status"] not in FINAL or tier in seen:
                    continue
                seen.add(tier)
                self._record_tier(job, item, tier, (time.perf_counter() - t0) * 1000)
            if all(item["status"] in FINAL for item in job["tasks"]):
                return
            time.sleep(POLL_SECONDS)
        _invalid("job %s did not finish in %ss" % (job_id[:8], JOB_TIMEOUT))

    @staticmethod
    def _record_tier(job, item, tier, client_ms):
        if item["status"] != "done":
            _invalid("NOT DONE %s (%s)" % (tier, item.get("error") or item["status"]))
            events.request.fire(request_type="TIER", name=tier + " (not done)",
                                response_time=client_ms, response_length=0, exception=None)
            return
        try:
            ms = (_parse(item["finished_at"]) - _parse(job["created_at"])).total_seconds() * 1000
            _TIMING["server"] += 1
        except (KeyError, TypeError, ValueError):
            ms = client_ms  # the status call had no timestamps: fall back, and say so
            _TIMING["client"] += 1
        events.request.fire(request_type="TIER", name=tier, response_time=ms,
                            response_length=0, exception=None)


@events.quitting.add_listener
def report(environment, **_kw):
    lines = ["MSP-41 load test", "hardware: %s" % hardware(),
             "jobs: %d finished of %d, users: %d" % (_COUNT["finished"], JOBS, _COUNT["users"]),
             "timing: %d tier results from server timestamps, %d from the client clock"
             % (_TIMING["server"], _TIMING["client"])]
    missed = False
    for tier, limit in P95_LIMIT_MS.items():
        entry = environment.stats.get(tier, "TIER")
        if not entry.num_requests:
            _invalid("NOT DONE %s (no tier ever finished done)" % tier)
            lines.append("%s p95: no samples" % tier)
            continue
        p95 = entry.get_response_time_percentile(0.95)
        ok = p95 < limit
        missed = missed or not ok
        lines.append("%s p95: %.0f ms (limit %.0f) %s over %d jobs"
                     % (tier, p95, limit, "PASS" if ok else "FAIL", entry.num_requests))
    if _COUNT["finished"] < JOBS:
        _invalid("only %d of %d jobs finished" % (_COUNT["finished"], JOBS))
    valid = not _INVALID
    if _INVALID:
        lines.append("RUN INVALID, do not use these numbers:")
        lines += ["  - %s x%d" % (reason, count) for reason, count in sorted(_INVALID.items())]
    else:
        lines.append("RUN VALID: no failed calls, no 429s, every tier done on every job")
    text = "\n".join(lines)
    print("\n" + text)
    path = RESULT_FILE if valid else RESULT_FILE.replace(".txt", "_INVALID.txt")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print("written to " + path)
    if missed or not valid:
        environment.process_exit_code = 1
