"""`/datasets` endpoints: upload, list, detail, delete.

Upload pipeline: sniff/size-check the file -> parse CSV defensively ->
`ml.validation.validate_transactions` -> persist the raw CSV (always) and the
cleaned Parquet (if valid) to Storage -> write the `datasets` row.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, UploadFile
from supabase import Client

from app import db, storage
from app.config import Settings, get_settings
from app.deps import CurrentUser, get_current_user, get_db_client, get_service_client
from app.errors import AppError
from app.logging import get_logger
from app.parquet_io import df_to_parquet_bytes, parquet_bytes_to_df
from app.schemas.datasets import DatasetRead, DatasetRowsResponse, DatasetUploadResponse, ValidationReportSchema
from app.schemas.jobs import ArtifactURL
from app.uploads import looks_like_csv, sanitize_filename
from ml.validation import ValidationConfig, validate_transactions

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = get_logger(__name__)

_ACCEPTED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
}

# Storage object paths are internal plumbing (read by the worker via
# `db.get_dataset`) and aren't part of the `DatasetRead` response schema.
_STORAGE_FIELDS = ("storage_path", "parquet_path")


def _to_dataset_read(row: dict) -> DatasetRead:
    return DatasetRead(**{k: v for k, v in row.items() if k not in _STORAGE_FIELDS})


@router.post("", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
    service_client: Client = Depends(get_service_client),
    settings: Settings = Depends(get_settings),
) -> DatasetUploadResponse:
    content_type = (file.content_type or "").lower()
    if content_type not in _ACCEPTED_CONTENT_TYPES:
        raise AppError("unsupported_media_type", "Only CSV uploads are supported", 415)

    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            "payload_too_large", f"File exceeds the {settings.max_upload_mb} MB limit", 413
        )
    if not data:
        raise AppError("bad_request", "Uploaded file is empty", 400)
    if not looks_like_csv(data):
        raise AppError("bad_request", "File does not appear to be a valid CSV", 400)

    try:
        raw_df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=True, na_filter=True)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise AppError("bad_request", f"Could not parse CSV: {exc}") from exc

    validation_config = ValidationConfig(
        min_transactions=settings.min_transactions,
        min_date_span_weeks=settings.min_date_span_weeks,
    )
    clean_df, report = validate_transactions(raw_df, validation_config)

    dataset_name = sanitize_filename(name or file.filename or "dataset.csv")
    dataset_row = db.create_dataset(db_client, current_user.org_id, dataset_name)
    dataset_id = dataset_row["id"]

    bucket = settings.supabase_storage_bucket
    raw_path = storage.raw_csv_path(current_user.org_id, dataset_id)
    storage.upload_bytes(service_client, bucket, raw_path, data, "text/csv")

    if not report.is_valid:
        db.update_dataset(
            db_client,
            dataset_id,
            current_user.org_id,
            storage_path=raw_path,
            status="failed",
            validation_report=report.to_dict(),
        )
        logger.info(
            "dataset_validation_failed",
            dataset_id=dataset_id,
            org_id=current_user.org_id,
            errors=report.errors,
        )
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            status="failed",
            validation_report=ValidationReportSchema(**report.to_dict()),
        )

    parquet_p = storage.parquet_path(current_user.org_id, dataset_id)
    storage.upload_bytes(
        service_client, bucket, parquet_p, df_to_parquet_bytes(clean_df), "application/octet-stream"
    )

    db.update_dataset(
        db_client,
        dataset_id,
        current_user.org_id,
        storage_path=raw_path,
        parquet_path=parquet_p,
        status="ready",
        validation_report=report.to_dict(),
        row_count=report.rows_kept,
        customer_count=report.distinct_customers,
        product_count=report.distinct_products,
        date_start=report.date_start.isoformat() if report.date_start else None,
        date_end=report.date_end.isoformat() if report.date_end else None,
    )

    logger.info(
        "dataset_ready",
        dataset_id=dataset_id,
        org_id=current_user.org_id,
        rows_kept=report.rows_kept,
        rows_dropped=report.rows_dropped,
    )

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        status="ready",
        validation_report=ValidationReportSchema(**report.to_dict()),
    )


@router.get("", response_model=list[DatasetRead])
async def list_datasets(
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> list[DatasetRead]:
    rows = db.list_datasets(db_client, current_user.org_id)
    return [_to_dataset_read(row) for row in rows]


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> DatasetRead:
    row = db.get_dataset(db_client, dataset_id, current_user.org_id)
    if row is None:
        raise AppError("not_found", "Dataset not found", 404)
    return _to_dataset_read(row)


@router.get("/{dataset_id}/rows", response_model=DatasetRowsResponse)
async def get_dataset_rows(
    dataset_id: str,
    page: int = 1,
    page_size: int = 50,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
    service_client: Client = Depends(get_service_client),
    settings: Settings = Depends(get_settings),
) -> DatasetRowsResponse:
    row = db.get_dataset(db_client, dataset_id, current_user.org_id)
    if row is None:
        raise AppError("not_found", "Dataset not found", 404)
    if row["status"] != "ready" or not row.get("parquet_path"):
        raise AppError("conflict", "Dataset is not ready", 409)

    page = max(1, page)
    page_size = max(1, min(500, page_size))

    data = storage.download_bytes(service_client, settings.supabase_storage_bucket, row["parquet_path"])
    df = parquet_bytes_to_df(data)

    total = len(df)
    start = (page - 1) * page_size
    slice_df = df.iloc[start : start + page_size]
    columns = list(slice_df.columns)
    rows = slice_df.where(slice_df.notna(), None).to_dict(orient="records")

    return DatasetRowsResponse(
        dataset_id=dataset_id,
        total=total,
        page=page,
        page_size=page_size,
        columns=columns,
        rows=rows,
    )


@router.get("/{dataset_id}/download", response_model=ArtifactURL)
async def download_dataset(
    dataset_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
    service_client: Client = Depends(get_service_client),
    settings: Settings = Depends(get_settings),
) -> ArtifactURL:
    row = db.get_dataset(db_client, dataset_id, current_user.org_id)
    if row is None:
        raise AppError("not_found", "Dataset not found", 404)
    if not row.get("storage_path"):
        raise AppError("conflict", "Dataset file is not available", 409)

    from datetime import UTC, datetime, timedelta

    url = storage.create_signed_url(service_client, settings.supabase_storage_bucket, row["storage_path"])
    expires_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    return ArtifactURL(url=url, expires_at=expires_at)


@router.delete("/{dataset_id}", status_code=204, response_model=None)
async def delete_dataset(
    dataset_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
    service_client: Client = Depends(get_service_client),
    settings: Settings = Depends(get_settings),
) -> None:
    row = db.get_dataset(db_client, dataset_id, current_user.org_id)
    if row is None:
        raise AppError("not_found", "Dataset not found", 404)

    paths = [p for p in (row.get("storage_path"), row.get("parquet_path")) if p]
    if paths:
        storage.delete_paths(service_client, settings.supabase_storage_bucket, paths)

    db.delete_dataset(db_client, dataset_id, current_user.org_id)
