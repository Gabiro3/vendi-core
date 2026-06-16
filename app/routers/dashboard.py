"""`GET /dashboard` — aggregated workspace summary for the frontend overview."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from supabase import Client

from app import db
from app.deps import CurrentUser, get_current_user, get_db_client
from app.schemas.dashboard import DashboardDatasets, DashboardJobs, DashboardResponse
from app.schemas.datasets import DatasetRead
from app.schemas.jobs import JobRead

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_STORAGE_FIELDS = ("storage_path", "parquet_path")


def _to_dataset_read(row: dict) -> DatasetRead:
    return DatasetRead(**{k: v for k, v in row.items() if k not in _STORAGE_FIELDS})


def _to_job_read(row: dict) -> JobRead:
    return JobRead(**row)


@router.get("", response_model=DashboardResponse)
async def get_dashboard(
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> DashboardResponse:
    datasets = db.list_datasets(db_client, current_user.org_id)
    all_jobs = db.list_jobs(db_client, current_user.org_id)

    ready = [d for d in datasets if d["status"] == "ready"]
    failed = [d for d in datasets if d["status"] == "failed"]
    validating = [d for d in datasets if d["status"] == "validating"]
    latest_dataset = _to_dataset_read(datasets[0]) if datasets else None

    active = [j for j in all_jobs if j["status"] in ("queued", "running")]
    recent = all_jobs[:5]

    latest_forecast = next(
        (j for j in all_jobs if j["module"] == "forecast" and j["status"] == "succeeded"), None
    )
    latest_basket = next(
        (j for j in all_jobs if j["module"] == "basket" and j["status"] == "succeeded"), None
    )
    latest_customer = next(
        (j for j in all_jobs if j["module"] == "customer" and j["status"] == "succeeded"), None
    )

    return DashboardResponse(
        datasets=DashboardDatasets(
            total=len(datasets),
            ready=len(ready),
            failed=len(failed),
            validating=len(validating),
            latest=latest_dataset,
        ),
        jobs=DashboardJobs(
            total=len(all_jobs),
            active=len(active),
            recent=[_to_job_read(j) for j in recent],
        ),
        latest_forecast=_to_job_read(latest_forecast) if latest_forecast else None,
        latest_basket=_to_job_read(latest_basket) if latest_basket else None,
        latest_customer=_to_job_read(latest_customer) if latest_customer else None,
    )
