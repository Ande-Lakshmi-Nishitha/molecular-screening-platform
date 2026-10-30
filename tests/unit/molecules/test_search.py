"""MSP-21 tests: mocks only, plus one live smoke test (run with MSP_LIVE=1)."""
import os
import unittest
from unittest import mock

from flask import Flask

from backend.api.errors import register_error_handling
from backend.api.molecules.search_routes import SERVICE_CONFIG_KEY, search_bp
from backend.services.search_client import pubchem
from backend.services.search_client.errors import UpstreamEmpty, UpstreamHTTPError, UpstreamTimeout
from backend.services.search_client.limiter import InProcessLimiter
from backend.services.search_client.service import SearchService

ASPIRIN = {"source": "pubchem", "id": "2244", "name": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
URL = "/api/molecules/search"


class FakeCache:
    def __init__(self):
        self.store, self.writes = {}, 0

    def get(self, source, query):
        return self.store.get((source, query))

    def set(self, source, query, payload):
        self.writes += 1
        self.store[(source, query)] = payload


class FakeLimiter:
    def __init__(self):
        self.calls = 0

    def acquire(self, service="PubChem", max_wait=5.0):
        self.calls += 1


def make_service(pubchem_fn):
    cache, limiter = FakeCache(), FakeLimiter()
    return SearchService(cache, limiter, {"pubchem": pubchem_fn, "chembl": lambda q: []}), cache, limiter


def raiser(exc):
    def fn(query):
        raise exc
    return fn


class ServiceTests(unittest.TestCase):
    def test_success_is_cached_and_second_call_skips_upstream(self):
        calls = []
        svc, _, limiter = make_service(lambda q: calls.append(q) or [ASPIRIN])
        first = svc.search("aspirin", "pubchem")
        second = svc.search("aspirin", "pubchem")
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(limiter.calls, 1)

    def test_empty_response_is_not_cached(self):
        svc, cache, _ = make_service(lambda q: [])
        with self.assertRaises(UpstreamEmpty) as ctx:
            svc.search("zzz", "pubchem")
        self.assertEqual(ctx.exception.service, "PubChem")
        self.assertEqual(cache.writes, 0)

    def test_timeout_and_http_error_are_not_cached(self):
        for exc in (UpstreamTimeout("PubChem"), UpstreamHTTPError("PubChem")):
            svc, cache, _ = make_service(raiser(exc))
            with self.assertRaises(type(exc)):
                svc.search("aspirin", "pubchem")
            self.assertEqual(cache.writes, 0)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        register_error_handling(self.app)
        self.app.register_blueprint(search_bp)
        # A live session: patch the two calls login_required makes.
        patches = [
            mock.patch("backend.api.auth.sessions.authenticate", return_value={"sub": "u1"}),
            mock.patch("backend.api.auth.sessions.get_session_store", return_value=object()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def get(self, pubchem_fn, url=f"{URL}?query=aspirin&source=pubchem"):
        svc, _, _ = make_service(pubchem_fn)
        self.app.config[SERVICE_CONFIG_KEY] = svc
        return self.app.test_client().get(url, headers={"Authorization": "Bearer x"})

    def test_success(self):
        res = self.get(lambda q: [ASPIRIN])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["results"][0]["id"], "2244")

    def test_timeout_is_504_and_names_service(self):
        res = self.get(raiser(UpstreamTimeout("PubChem", "PubChem timed out.")))
        body = res.get_json()["error"]
        self.assertEqual(res.status_code, 504)
        self.assertIn("PubChem", body["message"])
        self.assertEqual(body["details"][0]["service"], "PubChem")

    def test_http_error_is_502(self):
        res = self.get(raiser(UpstreamHTTPError("PubChem", "PubChem returned HTTP 503.")))
        self.assertEqual(res.status_code, 502)

    def test_empty_is_502(self):
        res = self.get(lambda q: [])
        self.assertEqual(res.status_code, 502)
        self.assertIn("PubChem", res.get_json()["error"]["message"])

    def test_bad_requests_are_400(self):
        self.assertEqual(self.get(lambda q: [ASPIRIN], URL).status_code, 400)
        self.assertEqual(self.get(lambda q: [ASPIRIN], f"{URL}?query=x&source=nope").status_code, 400)

    def test_no_token_is_401(self):
        with mock.patch("backend.api.auth.sessions.authenticate", side_effect=self._unauth()):
            res = self.app.test_client().get(f"{URL}?query=aspirin")
        self.assertEqual(res.status_code, 401)

    @staticmethod
    def _unauth():
        from backend.api.auth.service import AuthError
        return AuthError("UNAUTHENTICATED", "Authentication required.", 401)


class LimiterTests(unittest.TestCase):
    def test_sixth_request_in_one_second_waits(self):
        state = {"t": 0.0, "slept": []}

        def sleep(seconds):
            state["slept"].append(seconds)
            state["t"] += seconds

        limiter = InProcessLimiter(limit=5, window=1.0, clock=lambda: state["t"], sleep=sleep)
        for _ in range(5):
            limiter.acquire()
        self.assertEqual(state["slept"], [])
        limiter.acquire()
        self.assertGreaterEqual(sum(state["slept"]), 1.0 - 1e-9)


@unittest.skipUnless(os.getenv("MSP_LIVE") == "1", "live smoke test: set MSP_LIVE=1")
class LiveSmoke(unittest.TestCase):
    def test_pubchem_aspirin(self):
        results = pubchem.search_pubchem("aspirin")
        self.assertEqual(results[0]["inchi_key"], "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")


if __name__ == "__main__":
    unittest.main()
