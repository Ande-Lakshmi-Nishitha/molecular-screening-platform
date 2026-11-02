"""OpenAPI document served at /api/openapi.json (SRS section 3.4, MSP-47).

ENDPOINTS lists every route the app serves. tests/unit/test_openapi_matches_routes.py fails when a
route exists without an entry here, or an entry here has no route, so the document cannot drift.
The route functions are the source of truth; paths use {name} for path parameters.
"""
from flask import Blueprint, jsonify

openapi_bp = Blueprint("openapi", __name__)

ERROR_REF = {"$ref": "#/components/schemas/Error"}

# (path, method, summary, needs a bearer token)
ENDPOINTS = [
    ("/api/auth/register", "post", "Register a user", False),
    ("/api/auth/login", "post", "Log in and receive a session token", False),
    ("/api/auth/logout", "post", "Log out: the session token stops working", True),
    ("/api/molecules/search", "get", "Search PubChem or ChEMBL", True),
    ("/api/molecules/validate", "post", "Standardize a molecule; per-tier eligibility", True),
    ("/api/molecules/submit", "post", "Submit a molecule; returns a job with its tasks", True),
    ("/api/jobs", "get", "List the caller's own jobs (history)", True),
    ("/api/jobs/{job_id}", "get", "One job: per-tier state, partial results, summary", True),
    ("/api/jobs/{job_id}", "delete", "Cancel a queued or running job; results are kept", True),
    ("/api/jobs/{job_id}/report", "get", "Export a job's report as JSON or PDF", True),
    ("/api/targets", "get", "Supported targets", True),
    ("/api/models", "get", "Model information and validation metrics", True),
    ("/api/openapi.json", "get", "This document", False),
    ("/healthz", "get", "Liveness check", False),
    ("/readyz", "get", "Readiness: 200 when PostgreSQL and Redis answer, else 503", False),
]

# Details that differ from the default (2XX success, 400/500 errors). Only what the code does today.
DETAILS = {
    ("/api/molecules/submit", "post"): {"success": "202", "errors": ["422"]},
    ("/api/jobs/{job_id}", "get"): {"success": "200", "errors": ["404"]},
    ("/api/jobs/{job_id}", "delete"): {"success": "200", "errors": ["404", "409"]},
    ("/api/jobs/{job_id}/report", "get"): {"success": "200", "errors": ["404"]},
    ("/readyz", "get"): {"success": "200", "errors": ["503"]},
}
ERROR_TEXT = {
    "404": "Job not found",
    "409": "Every task of the job has already finished",
    "422": "Input rejected",
    "503": "A dependency is not ready",
}


def _error_response(description):
    return {
        "description": description,
        "content": {"application/json": {"schema": ERROR_REF}},
    }


def _operation(path, method, summary, needs_auth):
    details = DETAILS.get((path, method), {})
    responses = {
        details.get("success", "2XX"): {"description": "Success"},
        "400": _error_response("Invalid request"),
        "500": _error_response("Internal error"),
    }
    for code in details.get("errors", []):
        responses[code] = _error_response(ERROR_TEXT[code])
    operation = {"summary": summary, "responses": responses}
    if needs_auth:
        operation["security"] = [{"bearerAuth": []}]
        responses["401"] = _error_response("Not authenticated")
        responses["403"] = _error_response("Not permitted")
    if "{job_id}" in path:
        operation["parameters"] = [{"name": "job_id", "in": "path", "required": True,
                                    "schema": {"type": "string", "format": "uuid"}}]
    return operation


def build_spec():
    """Return the OpenAPI 3.0 document as a dict."""
    paths = {}
    for path, method, summary, needs_auth in ENDPOINTS:
        paths.setdefault(path, {})[method] = _operation(path, method, summary, needs_auth)
    return {
        "openapi": "3.0.3",
        "info": {"title": "Molecule Screening Platform API", "version": "1.0.0"},
        "paths": paths,
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {
                        "error": {
                            "type": "object",
                            "required": ["code", "message", "details", "request_id"],
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "details": {"description": "A list or an object"},
                                "request_id": {"type": "string"},
                            },
                        }
                    },
                }
            },
        },
    }


@openapi_bp.get("/api/openapi.json")
def openapi_json():
    """Serve the OpenAPI document."""
    return jsonify(build_spec())
