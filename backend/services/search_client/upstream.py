"""The only place that makes outgoing HTTP calls for search."""
import requests

from backend.services.search_client.errors import UpstreamEmpty, UpstreamHTTPError, UpstreamTimeout

TIMEOUT = (3, 10)  # connect, read (seconds)
MAX_RESULTS = 5


def get_json(service, url, params=None):
    """GET url and return parsed JSON, or raise an UpstreamError naming `service`."""
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT, headers={"Accept": "application/json"})
    except requests.Timeout:
        raise UpstreamTimeout(service, f"{service} timed out.") from None
    except requests.RequestException:
        raise UpstreamHTTPError(service, f"Could not reach {service}.") from None
    if resp.status_code == 404:
        raise UpstreamEmpty(service, f"{service} found no results for that query.")
    if resp.status_code >= 400:
        raise UpstreamHTTPError(service, f"{service} returned HTTP {resp.status_code}.")
    try:
        return resp.json()
    except ValueError:
        raise UpstreamEmpty(service, f"{service} returned an empty or unreadable response.") from None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
