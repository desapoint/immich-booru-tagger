FROM python:3.11-slim

LABEL org.opencontainers.image.title="Immich Booru Tagger"
LABEL org.opencontainers.image.description="AI-powered WD14/Booru tagging for Immich"
LABEL org.opencontainers.image.source="https://github.com/desapoint/immich-booru-tagger"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOME=/config \
    CONFIG_DIR=/config \
    XDG_CACHE_HOME=/config/cache \
    HF_HOME=/config/huggingface \
    MODEL_CACHE_DIR=/config/models

WORKDIR /app

# Runtime dependencies + helpers used by the Unraid-compatible entrypoint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        gosu \
        tini \
        passwd \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies separately so Docker can cache pip installs.
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source.
COPY . /app

# Ensure startup script is executable.
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh \
    && chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /config

# Never publish an image whose source fails to compile or pass its unit tests.
RUN python -m compileall -q /app/immich_tagger /app/tests \
    && python -m unittest discover -s /app/tests -v

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]

CMD ["python", "-m", "immich_tagger.main", "--mode", "scheduler"]
