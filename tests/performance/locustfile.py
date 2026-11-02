"""MSP-41 load test: 10 users, 20 queued jobs, ML p95 < 1 s, DL p95 < 3 s.

Traces to: NFR-001, NFR-002, NFR-003, NFR-012, R08. Run by hand against the reference stack
(never in CI), after `docker compose up`, with the real ML and DL models loaded and an existing account:

    set "MSP_EMAIL=loadtest@example.test"  &  set "MSP_PASSWORD=..."  &  set "MSP_TARGET_ID=EGFR"
    python -m locust -f tests/performance/locustfile.py --headless -u 10 -r 10 -t 10m --host https://localhost

Each simulated user logs in as its own account (loadtest-1@..., loadtest-2@..., registered on first use),
because the API allows each logged-in user RATE_LIMIT requests (60/minute by default). Set MSP_ONE_ACCOUNT=1 to
share the single MSP_EMAIL account instead.
The run stops itself once JOBS jobs have finished, or at the first failed login (the reason is printed).
Each tier is timed from submit until it is `done`; a tier that ends failed, skipped or cancelled is counted
separately and makes the run fail, because its time says nothing about the tier. The limits and the hardware
are open items (TBD-C10), so they come from the environment, not from code. The exit code is 1 if a limit
is missed, a tier did not finish, or a login failed.
"""
# pylint: disable=missing-function-docstring,too-few-public-methods
import os
import platform
import threading
import time
from datetime import datetime

import gevent
import urllib3
from locust import HttpUser, constant, events, task
from locust.exception import StopUser

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # self-signed certificate on the test stack

JOBS = int(os.environ.get("MSP_JOBS", "20"))
P95_LIMIT_MS = {"ML": float(os.environ.get("MSP_ML_P95_MS", "1000")),
                "DL": float(os.environ.get("MSP_DL_P95_MS", "3000"))}
EMAIL = os.environ.get("MSP_EMAIL", "")
PASSWORD = os.environ.get("MSP_PASSWORD", "")
SMILES = os.environ.get("MSP_SMILES", "CCO")
TARGET_ID = os.environ.get("MSP_TARGET_ID", "")       # needed by the DL tier
POLL_SECONDS = float(os.environ.get("MSP_POLL_SECONDS", "0.5"))
ONE_ACCOUNT = os.environ.get("MSP_ONE_ACCOUNT", "") == "1"  # default: one account per simulated user
JOB_TIMEOUT = float(os.environ.get("MSP_JOB_TIMEOUT", "120"))
RESULT_FILE = os.environ.get("MSP_RESULT_FILE", "tests/performance/last_run.txt")
FINAL = {"done", "failed", "skipped", "cancelled", "canceled"}
TOKEN_KEYS = ("token", "access_token", "accessToken")

_LOCK = threading.Lock()
_COUNT = {"started": 0, "finished": 0, "users": 0}
_PROBLEMS = []  # reasons the run is not valid (failed login, tiers that did not finish)


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


def _error_text(resp):
    """Code and message of an error body, never the request or any token."""
    try:
        error = resp.json().get("error", {})
        return "%s: %s" % (error.get("code", "?"), error.get("message", "?"))
    except (ValueError, AttributeError):
        return "no JSON body"


def _quit_run(environment):
    """Stop the whole run. It must not be called from a user's own greenlet (stopping the users would kill it
    before the run has quit), so it runs in a separate greenlet."""
    gevent.spawn(environment.runner.quit)


def _stop_run(environment, reason):
    with _LOCK:
        if reason in _PROBLEMS:  # ten users hit the same problem: say it once
            return
        _PROBLEMS.append(reason)
    print("\nSTOPPING: " + reason)
    environment.process_exit_code = 1
    _quit_run(environment)


def _tier_ms(created_at, finished_at, fallback_ms):
    """Milliseconds from the job being created to the task finishing, from the server's own timestamps.

    This does not depend on how often the client polls or on how slow the status call is. If a timestamp is
    missing or unreadable, the client's own clock is used instead."""
    try:
        return max((datetime.fromisoformat(finished_at) - datetime.fromisoformat(created_at)).total_seconds() * 1000, 0.0)
    except (TypeError, ValueError):
        return fallback_ms


class Researcher(HttpUser):
    wait_time = constant(0)

    def _account_email(self):
        """Each simulated user gets its own account, because the API limits each user to RATE_LIMIT requests."""
        with _LOCK:
            _COUNT["users"] += 1
            number = _COUNT["users"]
        local, _, domain = EMAIL.partition("@")
        return "%s-%d@%s" % (local, number, domain)

    def _login(self, email, expect_missing):
        """Returns (status, body, error text); body is None when the login failed."""
        with self.client.post("/api/auth/login", json={"email": email, "password": PASSWORD},
                              name="login", catch_response=True) as resp:
            if resp.status_code == 401 and expect_missing:
                resp.success()  # the account does not exist yet: it is registered next
                return 401, None, _error_text(resp)
            if not resp.ok:
                resp.failure("HTTP %s" % resp.status_code)
                return resp.status_code, None, _error_text(resp)
            return resp.status_code, resp.json(), ""

    def _register(self, email):
        with self.client.post("/api/auth/register", json={"email": email, "password": PASSWORD},
                              name="register", catch_response=True) as resp:
            if resp.status_code in (200, 201, 409):  # 409: another user got there first
                if resp.status_code == 409:
                    resp.success()
                return
            resp.failure("HTTP %s" % resp.status_code)
            _stop_run(self.environment, "registering %s failed: HTTP %s, %s. The password may break the "
                      "password rules." % (email, resp.status_code, _error_text(resp)))
            raise StopUser()

    def on_start(self):
        self.client.verify = False  # self-signed certificate on the test stack
        if not EMAIL or not PASSWORD or "@" not in EMAIL:
            _stop_run(self.environment, "MSP_EMAIL (an address) and MSP_PASSWORD are not set.")
            raise StopUser()
        email = EMAIL if ONE_ACCOUNT else self._account_email()
        status, body, why = self._login(email, expect_missing=not ONE_ACCOUNT)
        if status == 401 and not ONE_ACCOUNT:
            self._register(email)
            status, body, why = self._login(email, expect_missing=False)
        if body is None:
            _stop_run(self.environment, "login failed: HTTP %s, %s. Check the password and that the rate "
                      "limiter allows the logins." % (status, why))
            raise StopUser()
        token = next((body[key] for key in TOKEN_KEYS if body.get(key)), None)
        if not token:
            _stop_run(self.environment, "login worked but the response has none of %s (keys present: %s)."
                      % (", ".join(TOKEN_KEYS), ", ".join(sorted(body))))
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
                _quit_run(self.environment)

    def _get_job(self, job_id):
        """The job's status body, backing off on a rate limit; None if the job cannot be read."""
        for _ in range(5):
            with self.client.get("/api/jobs/" + job_id, name="/api/jobs/[id]", catch_response=True) as resp:
                if resp.status_code == 429:
                    resp.failure("rate limited")
                    time.sleep(1.0)
                    continue
                if not resp.ok:
                    resp.failure("HTTP %s" % resp.status_code)
                    return None
                return resp.json()
        return None

    def _run_job(self):
        body = {"smiles": SMILES}
        if TARGET_ID:
            body["target_id"] = TARGET_ID
        t0 = time.perf_counter()
        with self.client.post("/api/molecules/submit", json=body, name="submit", catch_response=True) as resp:
            if resp.status_code != 202:
                resp.failure("HTTP %s, %s" % (resp.status_code, _error_text(resp)))
                return
            job_id = resp.json()["job_id"]
        seen = set()
        while time.perf_counter() - t0 < JOB_TIMEOUT:
            job = self._get_job(job_id)
            if job is None:
                return
            tasks = job.get("tasks", [])
            now = (time.perf_counter() - t0) * 1000
            for item in tasks:
                tier = item["tier"].upper()
                if item["status"] in FINAL and tier not in seen:
                    seen.add(tier)
                    if item["status"] == "done":
                        # milliseconds from submit until the tier is done (server timestamps), as its own Locust row
                        took = _tier_ms(job.get("created_at"), item.get("finished_at"), now)
                        events.request.fire(request_type="TIER", name=tier, response_time=took,
                                            response_length=0, exception=None)
                    else:
                        events.request.fire(request_type="TIER", name="NOT DONE " + tier, response_time=now,
                                            response_length=0,
                                            exception=RuntimeError("%s ended %s (%s)" % (
                                                tier, item["status"], item.get("error") or item.get("skip_reason"))))
            if all(item["status"] in FINAL for item in tasks):
                return
            time.sleep(POLL_SECONDS)


@events.quitting.add_listener
def check_limits(environment, **_kw):
    users = getattr(getattr(environment, "parsed_options", None), "num_users", "?")
    lines = ["MSP-41 load test", "hardware: %s" % hardware(), "jobs: %d, users: %s" % (JOBS, users)]
    lines.append("timing: submit to tier done, from the server's timestamps (job created_at to task finished_at)")
    failed = bool(_PROBLEMS)
    lines += ["PROBLEM: " + problem for problem in _PROBLEMS]
    for (name, method), entry in sorted(environment.stats.entries.items()):
        if method != "TIER" and entry.num_failures:
            lines.append("PROBLEM: %s %s failed %d time(s), so the run is not clean" % (method, name, entry.num_failures))
            failed = True
    for tier, limit in P95_LIMIT_MS.items():
        entry = environment.stats.get(tier, "TIER")
        if not entry.num_requests:
            lines.append("%s p95: no samples (the tier never finished 'done')" % tier)
            failed = True
            continue
        p95 = entry.get_response_time_percentile(0.95)
        ok = p95 < limit
        failed = failed or not ok
        lines.append("%s p95: %.0f ms over %d jobs (limit %.0f) %s" % (
            tier, p95, entry.num_requests, limit, "PASS" if ok else "FAIL"))
    not_done = {name: entry.num_requests for (name, method), entry in environment.stats.entries.items()
                if method == "TIER" and name.startswith("NOT DONE")}
    for name, count in sorted(not_done.items()):
        lines.append("%s: %d times, so the timing is not valid" % (name, count))
        failed = True
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(os.path.dirname(RESULT_FILE) or ".", exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    if failed:
        environment.process_exit_code = 1
