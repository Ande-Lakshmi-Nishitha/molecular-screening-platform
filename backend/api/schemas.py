"""Strict request validation (SEC-004).

Subclass StrictModel for each request body. Unknown fields and wrong types
are rejected. Error details name the field and the problem only, never the
submitted value, so molecules never come back in an error response.
"""
from flask import request
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.api.errors import ApiError


class StrictModel(BaseModel):
    """Base class for request bodies: no extra fields, no type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


def validate_body(model_cls):
    """Parse the JSON body with model_cls or raise a 400 ApiError."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_json", "Request body must be a JSON object.")
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        details = [
            {"field": ".".join(str(p) for p in err["loc"]) or "body", "type": err["type"]}
            for err in exc.errors(include_url=False, include_context=False, include_input=False)
        ]
        raise ApiError(
            400, "validation_error", "Request body failed validation.", details
        ) from None
