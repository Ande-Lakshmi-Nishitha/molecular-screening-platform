"""WSGI entry point for gunicorn."""
from backend.api.app import create_app

app = create_app()
