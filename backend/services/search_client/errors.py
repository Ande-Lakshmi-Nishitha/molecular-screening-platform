"""Upstream failures for PubChem / ChEMBL search (MSP-21; FR-005, FR-006, FR-008).

Every error names the failing service. Pure Python: no Flask, no Redis. The route
turns these into the standard ApiError envelope.
"""


class UpstreamError(Exception):
    status = 502
    code = "UPSTREAM_ERROR"
    kind = "http_error"

    def __init__(self, service, message=None):
        self.service = service
        super().__init__(message or f"{service} request failed.")


class UpstreamTimeout(UpstreamError):
    status = 504
    code = "UPSTREAM_TIMEOUT"
    kind = "timeout"


class UpstreamHTTPError(UpstreamError):
    status = 502
    code = "UPSTREAM_ERROR"
    kind = "http_error"


class UpstreamEmpty(UpstreamError):
    status = 502
    code = "UPSTREAM_EMPTY"
    kind = "empty_response"


class RateLimited(UpstreamError):
    status = 429
    code = "TOO_MANY_REQUESTS"
    kind = "rate_limited"
