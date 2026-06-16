"""`/jobs` endpoints: list, status, and signed-URL artifact download."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from supabase import Client

from app import db, storage
from app.config import Settings, get_settings
from app.deps import CurrentUser, get_current_user, get_db_client, get_service_client
from app.errors import AppError
from app.schemas.jobs import ArtifactURL, JobRead, JobStatus, ModuleName

router = APIRouter(prefix="/jobs", tags=["jobs"])

_ARTIFACT_URL_EXPIRES_IN = 3600


@router.get("", response_model=list[JobRead])
async def list_jobs(
    module: ModuleName | None = Query(default=None),
    status: JobStatus | None = Query(default=None),
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> list[JobRead]:
    rows = db.list_jobs(db_client, current_user.org_id, module=module, status=status)
    return [JobRead(**row) for row in rows]


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> JobRead:
    row = db.get_job(db_client, job_id, current_user.org_id)
    if row is None:
        raise AppError("not_found", "Job not found", 404)
    return JobRead(**row)


@router.get("/{job_id}/artifact", response_model=ArtifactURL)
async def get_job_artifact(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
    service_client: Client = Depends(get_service_client),
    settings: Settings = Depends(get_settings),
) -> ArtifactURL:
    job = db.get_job(db_client, job_id, current_user.org_id)
    if job is None:
        raise AppError("not_found", "Job not found", 404)
    if not job.get("result_storage_path"):
        raise AppError("not_found", "This job has no downloadable artifact", 404)

    url = storage.create_signed_url(
        service_client,
        settings.supabase_storage_bucket,
        job["result_storage_path"],
        expires_in=_ARTIFACT_URL_EXPIRES_IN,
    )
    expires_at = datetime.now(UTC) + timedelta(seconds=_ARTIFACT_URL_EXPIRES_IN)
    return ArtifactURL(url=url, expires_at=expires_at)
