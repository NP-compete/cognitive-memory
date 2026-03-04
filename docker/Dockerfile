# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

RUN pip install --no-cache-dir build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m build --wheel

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /app/dist/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl[all] && \
    rm /tmp/*.whl

USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "cognitive_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
