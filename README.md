# Vendi — Retail Intelligence Platform (Backend)

Backend API and ML modules for Vendi, a retail intelligence platform for
mid-to-large retailers (supermarkets, shopping centers, etc.). It ingests
transaction-level sales data and runs three independent analytics modules:

- **Demand forecasting** (LightGBM, per product/store, with prediction intervals)
- **Market-basket analysis** (FP-Growth/mlxtend association rules)
- **Customer modeling** (BG/NBD + Gamma-Gamma CLV, iso-value segmentation)

## Architecture

```
app/    FastAPI service - all I/O (Supabase Postgres/Storage/Auth, ARQ/Redis job queue)
ml/     Pure analytics modules - DataFrame in, typed dataclass result out
        No module under ml/ imports from app/, and none of it talks to the
        network, a database, or storage. app/ owns all I/O.
migrations/   SQL migrations (tables + Row-Level Security policies)
tests/  pytest suite: tests/ml (pure module tests), tests/api (FastAPI
        integration tests against an in-memory fake Supabase client)
```

### `app/`

- `main.py` — FastAPI app factory, middleware, exception handlers, router wiring
- `config.py` — `pydantic-settings` config loaded from environment / `.env`
- `deps.py` — auth (`get_current_user`), Supabase clients, Redis pool
- `security.py` — JWT verification (Supabase JWT secret or JWKS)
- `db.py` — Postgrest query helpers (every query explicitly filters by `org_id`)
- `storage.py` — Supabase Storage helpers (signed URLs, server-generated paths)
- `parquet_io.py` — DataFrame <-> Parquet (Storage) helpers
- `jobs.py` / `worker.py` / `run_helpers.py` — ARQ job queue + worker entrypoint;
  `SYNC_JOBS=true` runs jobs in-process (no Redis) for local/dev use
- `routers/` — `datasets`, `forecast`, `basket`, `customer`, `jobs`, `me`
- `schemas/` — Pydantic request/response models (`extra="forbid"`)

### `ml/`

- `contracts.py` — single source of truth for all ML input/output dataclasses
- `validation.py` — header normalization + transaction validation/cleaning
- `features.py` — shared feature engineering helpers
- `forecasting/` — `model.py` (LightGBM forecast + recursive multi-step),
  `evaluate.py` (backtesting, feature importances)
- `basket/rules.py` — FP-Growth association rules, optional dimension slicing
- `customer/clv.py` — BG/NBD + Gamma-Gamma CLV
- `customer/isovalue.py` — value segmentation + iso-value contour grid

## Setup

Requires Python 3.11+ (developed/tested on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -e ".[dev]"
cp .env.example .env           # fill in Supabase credentials, etc.
```

On Windows, activate with `.venv\Scripts\activate.bat` (cmd.exe) or
`.venv\Scripts\Activate.ps1` (PowerShell — requires
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` if script execution is
disabled). Or just call `.venv\Scripts\python.exe` / `.venv\Scripts\pip.exe`
directly without activating.

### Database

Apply `migrations/0001_init.sql` to your Supabase Postgres project (via the
SQL editor or `psql`). It creates `organizations`, `memberships`, `datasets`,
`jobs`, RLS policies, and the `updated_at` trigger.

### Running the API

```bash
uvicorn app.main:app --reload
```

For local development without Redis, set `SYNC_JOBS=true` in `.env` — module
jobs run synchronously in-process and are `succeeded`/`failed` by the time
`POST /<module>/run` returns.

### Running the worker (production / `SYNC_JOBS=false`)

```bash
arq app.worker.WorkerSettings
```

## API overview

All endpoints (except `/health`) require `Authorization: Bearer <supabase-jwt>`
and are scoped to the caller's organization.

- `GET /me` — current user, org, role, and memberships
- `POST /datasets` — upload a CSV, validate, and persist (raw + cleaned Parquet)
- `GET /datasets`, `GET /datasets/{id}`, `DELETE /datasets/{id}`
- `POST /forecast/run`, `GET /forecast/{job_id}/results`
- `POST /basket/run`, `GET /basket/{job_id}/results`
- `POST /customer/run`, `GET /customer/{job_id}/results`
- `GET /jobs`, `GET /jobs/{job_id}`, `GET /jobs/{job_id}/artifact`

Module `run` endpoints are idempotent: re-running with the same dataset and
params returns the existing succeeded job unless `force: true` is passed.

## Testing

```bash
pytest                 # full suite (ml + api)
pytest tests/ml        # pure ML module tests (no I/O)
pytest tests/api       # FastAPI integration tests (fake Supabase client)
```

`tests/api` uses an in-memory fake Supabase Postgrest/Storage client
(`tests/api/fakes.py`) and FastAPI `dependency_overrides`, so the full suite
runs without a real Supabase project, Redis, or network access.

## Linting / type checking

```bash
ruff check .
black --check .
mypy app ml
```
