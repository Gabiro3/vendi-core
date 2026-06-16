"""Schemas for `GET /dashboard`."""

from __future__ import annotations

from app.schemas.common import APIModel
from app.schemas.datasets import DatasetRead
from app.schemas.jobs import JobRead


class DashboardDatasets(APIModel):
    total: int
    ready: int
    failed: int
    validating: int
    latest: DatasetRead | None


class DashboardJobs(APIModel):
    total: int
    active: int
    recent: list[JobRead]


class DashboardResponse(APIModel):
    datasets: DashboardDatasets
    jobs: DashboardJobs
    latest_forecast: JobRead | None
    latest_basket: JobRead | None
    latest_customer: JobRead | None
