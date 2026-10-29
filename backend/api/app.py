"""Flask application factory (MSP-14).

Every blueprint is registered in register_blueprints(): add only your own one
line there, and pull main before editing.
"""
import os

from flask import Flask, request

from backend.api.errors import register_error_handling
from backend.api.health import health_bp
from backend.api.molecules.validate_routes import validate_bp
from backend.api.openapi import openapi_bp
from backend.core.logging_filter import install_log_filter

DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
DEFAULT_FRONTEND_ORIGIN = "https://localhost"


def register_blueprints(app):
    """Register every API blueprint. One line per blueprint."""
    app.register_blueprint(health_bp)
    app.register_blueprint(openapi_bp)
    app.register_blueprint(validate_bp)


def _install_cors(app):
    """Allow cross-origin calls from the frontend origin only."""

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin and origin == app.config["FRONTEND_ORIGIN"]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = (
                "Authorization, Content-Type, X-Request-ID"
            )
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            response.headers["Access-Control-Expose-Headers"] = "X-Request-ID"
        response.vary.add("Origin")
        return response


def create_app(config=None):
    """Build the Flask app. config overrides defaults (used by tests)."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES)
    )
    app.config["FRONTEND_ORIGIN"] = os.environ.get("FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN)
    if config:
        app.config.update(config)
    install_log_filter()
    register_error_handling(app)
    _install_cors(app)
    register_blueprints(app)
    return app
