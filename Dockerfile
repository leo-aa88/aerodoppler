# AeroDoppler: API + built UI in one container. Binds to 127.0.0.1 inside the container; map it
# to a localhost port only (see docker-compose.yml). No data leaves the container unless an
# ADSB.lol / DEM import is requested.
FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS app
ENV PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1 AERODOPPLER_DATA_DIR=/data AERODOPPLER_HOST=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system .
COPY --from=web /web/out ./web/out
COPY Makefile SPEC.md IMPLEMENTATION_REPORT.md ./
VOLUME ["/data"]
EXPOSE 8000
# 0.0.0.0 is required inside the container; docker-compose publishes it on 127.0.0.1 only.
CMD ["aerodoppler", "serve", "--host", "0.0.0.0", "--port", "8000"]
