# Build context is the repo root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY deploy/worker_entrypoint.sh deploy/worker_healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/worker_entrypoint.sh /usr/local/bin/worker_healthcheck.sh \
    && useradd --create-home app
USER app

EXPOSE 8000
CMD ["sh", "-c", "exec gunicorn --workers ${GUNICORN_WORKERS:-2} --bind 0.0.0.0:8000 backend.wsgi:app"]