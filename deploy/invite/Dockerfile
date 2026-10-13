# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS mobile-build
WORKDIR /build/mobile
COPY clients/mobile/package.json clients/mobile/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY clients/mobile/index.html clients/mobile/vite.config.js ./
COPY clients/mobile/src ./src
COPY clients/mobile/public ./public
COPY clients/mobile/scripts ./scripts
RUN npm run build:h5

FROM python:3.11-slim-bookworm AS python-build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY deploy/invite/requirements.txt /build/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes -r /build/requirements.txt

FROM python:3.11-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHON_DOTENV_DISABLED=1 \
    DOCTHINKER_INVITE_DATA_DIR=/var/lib/docthinker \
    DOCTHINKER_INVITE_WEB_DIR=/app/web \
    DOCTHINKER_INVITE_BIND_HOST=0.0.0.0 \
    DOCTHINKER_INVITE_PORT=8080
RUN groupadd --gid 10001 docthinker \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin docthinker \
    && install -d -o 10001 -g 10001 -m 0700 /var/lib/docthinker
WORKDIR /app
COPY --from=python-build /opt/venv /opt/venv
COPY docthinker ./docthinker
COPY graphcore ./graphcore
COPY --from=mobile-build /build/mobile/dist/build/h5/index.html /build/mobile/dist/build/h5/manifest.webmanifest /build/mobile/dist/build/h5/pwa-register.js /build/mobile/dist/build/h5/service-worker.js ./web/
COPY --from=mobile-build /build/mobile/dist/build/h5/assets ./web/assets
COPY --from=mobile-build /build/mobile/dist/build/h5/static ./web/static
# Import smoke test only: no lifespan, stores, providers or network requests.
RUN python -c "from docthinker.server.mobile_app import create_mobile_app; from docthinker.server.invite_runtime import InviteRuntime; from docthinker.harness_runtime import HarnessRun; from docthinker.server.insight_runtime import InsightRuntime; from docthinker.server.knowledge_skills import knowledge_skill_catalog, _load_pack; assert all(_load_pack(item['id']) for item in knowledge_skill_catalog()); create_mobile_app()"
USER 10001:10001
EXPOSE 8080
CMD ["python", "-m", "docthinker.server.invite_runtime"]
