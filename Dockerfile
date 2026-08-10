# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.13-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=Asia/Shanghai

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        fontconfig \
        fonts-noto-cjk \
        libgomp1 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-install-project \
    && groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home app

USER app

CMD ["python", "-m", "src.business", "serve"]
