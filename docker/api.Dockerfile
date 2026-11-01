# The prediction API. The model and warehouse are mounted at run time (see
# docker-compose.yml), so the image holds code and dependencies only.
FROM python:3.11-slim-bookworm

# XGBoost needs OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.13.0

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src

RUN useradd --create-home --uid 10001 api \
    && mkdir logs \
    && chown api logs
USER api

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"]
CMD ["uvicorn", "opspilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
