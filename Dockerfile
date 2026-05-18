FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv build --wheel --out-dir dist/


# --- Final stage ---
FROM python:3.14-slim-bookworm

WORKDIR /app

COPY --from=builder /app/dist/*.whl ./

RUN pip install --no-cache-dir *.whl && rm *.whl

CMD ["python3", "-m", "notes_rag"]
