"""Schemas for `/datasets` endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from app.schemas.common import APIModel

DatasetStatus = Literal["validating", "ready", "failed"]


class ValidationReportSchema(APIModel):
    rows_in: int
    rows_kept: int
    rows_dropped: int
    dropped_by_reason: dict[str, int]
    duplicate_rows_removed: int
    date_start: date | None
    date_end: date | None
    distinct_transactions: int
    distinct_products: int
    distinct_customers: int | None
    distinct_stores: int | None
    has_customer_id: bool
    has_category: bool
    has_store_id: bool
    warnings: list[str]
    errors: list[str]


class DatasetRead(APIModel):
    id: str
    org_id: str
    name: str
    status: DatasetStatus
    validation_report: ValidationReportSchema | None = None
    row_count: int | None = None
    customer_count: int | None = None
    product_count: int | None = None
    date_start: date | None = None
    date_end: date | None = None
    created_at: datetime


class DatasetUploadResponse(APIModel):
    dataset_id: str
    status: DatasetStatus
    validation_report: ValidationReportSchema


class DatasetRowsResponse(APIModel):
    dataset_id: str
    total: int
    page: int
    page_size: int
    columns: list[str]
    rows: list[dict]
