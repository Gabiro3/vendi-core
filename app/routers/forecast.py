"""`/forecast/*` endpoints."""

from __future__ import annotations

from arq import ArqRedis
from fastapi import APIRouter, Depends
from supabase import Client

from app import db
from app.config import Settings, get_settings
from app.deps import (
    CurrentUser,
    get_current_user,
    get_db_client,
    get_redis_pool,
    get_service_client,
)
from app.errors import AppError
from app.run_helpers import start_module_job
from app.schemas.forecast import ForecastResultSchema, ForecastRunRequest
from app.schemas.jobs import JobAccepted

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("/run", response_model=JobAccepted, status_code=202)
async def run_forecast(
    request: ForecastRunRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
    service_client: Client = Depends(get_service_client),
    redis_pool: ArqRedis | None = Depends(get_redis_pool),
    settings: Settings = Depends(get_settings),
) -> JobAccepted:
    return await start_module_job(
        module="forecast",
        dataset_id=request.dataset_id,
        params=request.params.model_dump(),
        force=request.force,
        current_user=current_user,
        db_client=db_client,
        service_client=service_client,
        redis_pool=redis_pool,
        settings=settings,
    )


@router.get("/{job_id}/results", response_model=ForecastResultSchema)
async def get_forecast_results(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> ForecastResultSchema:
    job = db.get_job(db_client, job_id, current_user.org_id)
    if job is None or job["module"] != "forecast":
        raise AppError("not_found", "Job not found", 404)
    if job["status"] != "succeeded":
        raise AppError("conflict", f"Job is not finished (status={job['status']})", 409)
    return ForecastResultSchema(**job["result_summary"])
