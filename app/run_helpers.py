"""Shared logic for `POST /<module>/run` endpoints.

Each `/forecast/run`, `/basket/run`, `/customer/run` endpoint validates its
own params shape, then defers to `start_module_job` for the common dataset
readiness check, idempotency lookup, job creation, and dispatch.
"""

from __future__ import annotations

from typing import Any

from arq import ArqRedis
from supabase import Client

from app import db
from app.config import Settings
from app.deps import CurrentUser
from app.errors import AppError
from app.jobs import enqueue_module_job
from app.schemas.jobs import JobAccepted
from app.worker import process_job


async def start_module_job(
    *,
    module: str,
    dataset_id: str,
    params: dict[str, Any],
    force: bool,
    current_user: CurrentUser,
    db_client: Client,
    service_client: Client,
    redis_pool: ArqRedis | None,
    settings: Settings,
) -> JobAccepted:
    dataset = db.get_dataset(db_client, dataset_id, current_user.org_id)
    if dataset is None:
        raise AppError("not_found", "Dataset not found", 404)
    if dataset["status"] != "ready":
        raise AppError(
            "conflict", f"Dataset is not ready (status={dataset['status']})", 409
        )

    if not force:
        existing = db.find_existing_succeeded_job(
            db_client, current_user.org_id, dataset_id, module, params
        )
        if existing is not None:
            return JobAccepted(job_id=existing["id"], status=existing["status"])

    job = db.create_job(db_client, current_user.org_id, dataset_id, module, params)
    job_id = job["id"]
    status = job["status"]

    if settings.sync_jobs:
        await process_job(service_client, job_id)
        job = db.get_job(db_client, job_id, current_user.org_id)
        status = job["status"] if job else status
    else:
        assert redis_pool is not None  # guaranteed when sync_jobs is False (see app.deps)
        await enqueue_module_job(redis_pool, job_id, module)

    return JobAccepted(job_id=job_id, status=status)
