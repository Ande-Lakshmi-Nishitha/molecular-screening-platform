"""Error envelope and request IDs (SEC-004, SEC-007).

Every error response has the shape
{"error": {"code", "message", "details", "request_id"}} and every response,
success or error, carries an X-Request-ID header. Unhandled exceptions return
a generic 500: no stack trace, SQL or exception text reaches the client.
"""
import logging
import re
import uuid

from flask import g, jsonify, request
from werkzeug.exceptions import HTTPException

LOGGER = logging.getLogger("msp.api")
REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    429: "too_many_requests",
}
GENERIC_500 = "An internal error occurred."


class ApiError(Exception):
    """Raise anywhere in a route to return the standard error envelope."""

    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or []


def current_request_id():
    """Request ID of the current request (created if missing)."""
    request_id = g.get("request_id")
    if request_id is None:
        request_id = uuid.uuid4().hex
        g.request_id = request_id
    return request_id


def error_response(status, code, message, details=None):
    """Build the standard error response."""
    body = {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
            "request_id": current_request_id(),
        }
    }
    response = jsonify(body)
    response.status_code = status
    return response


def _handle_api_error(exc):
    return error_response(exc.status, exc.code, exc.message, exc.details)


def _handle_http_exception(exc):
    status = exc.code or 500
    if status >= 500:
        return error_response(status, "internal_error", GENERIC_500)
    code = STATUS_CODES.get(status, "request_error")
    response = error_response(status, code, exc.description)
    for name, value in exc.get_headers():
        if name.lower() in ("allow", "retry-after"):
            response.headers[name] = value
    return response


def _handle_unexpected(exc):
    LOGGER.error(
        "unhandled %s request_id=%s", type(exc).__name__, current_request_id(), exc_info=exc
    )
    return error_response(500, "internal_error", GENERIC_500)


def _before_request():
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    if _VALID_REQUEST_ID.match(incoming):
        g.request_id = incoming
    else:
        g.request_id = uuid.uuid4().hex


def _after_request(response):
    response.headers[REQUEST_ID_HEADER] = current_request_id()
    return response


def register_error_handling(app):
    """Attach request-ID hooks and all error handlers to the app."""
    app.before_request(_before_request)
    app.after_request(_after_request)
    app.register_error_handler(ApiError, _handle_api_error)
    app.register_error_handler(HTTPException, _handle_http_exception)
    app.register_error_handler(Exception, _handle_unexpected)
