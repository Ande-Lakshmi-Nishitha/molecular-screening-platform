"""WSGI entry point for gunicorn.

MSP-11 placeholder: serves only the health blueprint. MSP-14 replaces this
with the real application factory in backend/api/app.py.
"""
from flask import Flask

from backend.api.health import health_bp

app = Flask(__name__)
app.register_blueprint(health_bp)
