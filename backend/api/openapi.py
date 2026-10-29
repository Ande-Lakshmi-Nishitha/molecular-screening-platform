"""OpenAPI document served at /api/openapi.json (SRS section 3.4).

Paths below are the SRS endpoint list. Each story that builds an endpoint adds
its request and response schemas here as it goes.
"""
from flask import Blueprint, jsonify

openapi_bp = Blueprint("openapi", __name__)

ERROR_REF = {"$ref": "#/components/schemas/Error"}

ENDPOINTS = [
    ("/api/auth/register", "post", "Register a user", False),
    ("/api/auth/login", "post", "Log in", False),
    ("/api/molecules/search", "get", "Search PubChem or ChEMBL", True),
    ("/api/molecules/validate", "post", "Standardize a molecule", True),
    ("/api/molecules/submit", "post", "Submit a molecule and receive a job ID", True),
    ("/api/jobs/{id}/status", "get", "Per-tier task state", True),
    ("/api/jobs/{id}/results", "get", "Per-tier results and triage summary", True),
    ("/api/jobs/{id}", "delete", "Cancel a job", True),
    ("/api/targets", "get", "Supported targets", True),
    ("/api/models", "get", "Model information and validation metrics", True),
    ("/healthz", "get", "Liveness check", False),
    ("/readyz", "get", "Readiness check", False),
]


def _error_response(description):
    return {
        "description": description,
        "content": {"application/json": {"schema": ERROR_REF}},
    }


def build_spec():
    """Return the OpenAPI 3.0 document as a dict."""
    paths = {}
    for path, method, summary, needs_auth in ENDPOINTS:
        operation = {
            "summary": summary,
            "responses": {
                "200": {"description": "Success"},
                "400": _error_response("Invalid request"),
                "500": _error_response("Internal error"),
            },
        }
        if needs_auth:
            operation["security"] = [{"bearerAuth": []}]
            operation["responses"]["401"] = _error_response("Not authenticated")
            operation["responses"]["403"] = _error_response("Not permitted")
        paths.setdefault(path, {})[method] = operation
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
                                "details": {"type": "array", "items": {"type": "object"}},
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
