# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
#   Installs all Python dependencies (including heavy build-time ones like gcc)
#   into an isolated prefix so they can be copied to the lean runtime stage.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# gcc/g++ cover any source-only transitive deps.
# LightGBM ships pre-built wheels for Python 3.11+ on manylinux so it
# typically doesn't need compilation, but the toolchain is cheap insurance.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what pip needs to resolve and install the package.
COPY pyproject.toml .
COPY app/ app/
COPY ml/ ml/

# --prefix keeps installed files separate from the builder's own Python so the
# COPY in the next stage is a clean, explicit transfer.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime (shared by API and worker)
#
# Override CMD when running the worker:
#   docker run <image> python -m arq app.worker:WorkerSettings
# Or in docker-compose set `command:` on the worker service (see docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# libgomp1  — OpenMP runtime required by LightGBM at import time even when
#             the wheel was pre-compiled (links to libgomp dynamically).
# ca-certificates — lets the Supabase client verify TLS to *.supabase.co.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring in the installed packages (app, ml, and all third-party deps).
# /install mirrors the layout of /usr/local so bin/ and lib/ land in the
# right places for Python's default sys.path.
COPY --from=builder /install /usr/local

# ── non-root user ─────────────────────────────────────────────────────────────
# Running as root inside a container is an unnecessary risk.
# The app code lives in site-packages (copied above) so no extra permissions
# on /home/app are required.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser

# ── environment defaults ──────────────────────────────────────────────────────
# These are sensible production defaults; override via env_file / -e flags.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=prod \
    SYNC_JOBS=false

# ── health check (API only) ───────────────────────────────────────────────────
# The /health endpoint returns {"status":"ok"} with no auth required.
# The worker service can override HEALTHCHECK to a Redis ping via compose.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=4); \
         sys.exit(0 if r.status == 200 else 1)"

# ── default command: API server ───────────────────────────────────────────────
# Workers: default 2, tune with WEB_CONCURRENCY.
# Timeout 120 s covers slow dataset validation on upload (sync, in-process).
# ML jobs run in the ARQ worker, so typical API responses are fast.
CMD ["sh", "-c", \
     "gunicorn app.main:app \
      --worker-class uvicorn.workers.UvicornWorker \
      --workers ${WEB_CONCURRENCY:-2} \
      --bind 0.0.0.0:8000 \
      --timeout 120 \
      --access-logfile - \
      --error-logfile -"]
