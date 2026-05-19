# syntax=docker/dockerfile:1.7
# --- builder ---
FROM python:3.14-slim-bookworm AS builder

SHELL ["sh", "-exc"]

ENV DEBIAN_FRONTEND=noninteractive
RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    build-essential \
    gcc
EOT

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_PYTHON=python3.14

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --no-install-project --no-dev

COPY . /src/

WORKDIR /src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- runtime ---
FROM python:3.14-slim-bookworm AS runtime

RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    git \
    openssh-client \
    ca-certificates

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOT

ENV PATH="/app/bin:$PATH"

RUN <<EOT
groupadd -r --gid 1000 app
useradd -r -d /app --uid 1000 --gid app -N app
EOT

COPY --from=builder --chown=app:app /app /app

USER app
WORKDIR /app

EXPOSE 9099

ENTRYPOINT ["python3", "-m", "notes_rag"]
