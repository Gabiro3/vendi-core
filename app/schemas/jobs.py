"""Schemas for the `jobs` table / `/jobs` endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from app.schemas.common import APIModel

ModuleName = Literal["forecast", "basket", "customer"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class JobRead(APIModel):
    id: str
    org_id: str
    dataset_id: str
    module: ModuleName
    status: JobStatus
    params: dict[str, Any]
    result_summary: dict[str, Any] | None = None
    result_storage_path: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class JobAccepted(APIModel):
    job_id: str
    status: JobStatus


class ArtifactURL(APIModel):
    url: str
    expires_at: datetime
