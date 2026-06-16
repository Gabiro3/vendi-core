"""ARQ worker: task functions that load a dataset, run the corresponding pure
`ml/` module, and persist results.

CPU-bound `ml/` calls run synchronously within the (async) task function. Per
TECHNICAL_SPEC.md §8, offloading to a process pool is a hardening item; v1
relies on ARQ's per-worker job concurrency (`max_jobs`) to bound load.

`process_job` is the shared core, also called directly (no Redis) when
`SYNC_JOBS=true` for local smoke tests - see `app/run_helpers.py`.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from arq.connections import RedisSettings
from supabase import Client

from app import db, storage
from app.config import get_settings
from app.deps import get_service_client
from app.logging import configure_logging, get_logger
from app.parquet_io import parquet_bytes_to_df
from app.schemas.basket import BasketResultSchema
from app.schemas.customer import RECORDS_PREVIEW_LIMIT, CustomerResultSchema
from app.schemas.forecast import ForecastResultSchema
from app.serialization import to_jsonable
from ml.basket import analyze_baskets
from ml.contracts import BasketParams, CustomerParams, ForecastParams
from ml.customer import model_customers
from ml.forecasting import forecast

logger = get_logger(__name__)


async def process_job(client: Client, job_id: str) -> None:
    job = db.get_job_by_id(client, job_id)
    if job is None:
        logger.error("job_not_found", job_id=job_id)
        return

    org_id, dataset_id, module = job["org_id"], job["dataset_id"], job["module"]
    settings = get_settings()
    db.update_job(client, job_id, org_id, status="running", started_at=_now())

    try:
        dataset = db.get_dataset(client, dataset_id, org_id)
        if dataset is None or not dataset.get("parquet_path"):
            raise ValueError("Dataset is not ready (missing processed data)")

        data = storage.download_bytes(
            client, settings.supabase_storage_bucket, dataset["parquet_path"]
        )
        transactions = parquet_bytes_to_df(data)

        result_summary, artifact = _run_module(module, transactions, job["params"])

        result_storage_path = None
        if artifact is not None:
            filename, content = artifact
            result_storage_path = storage.job_artifact_path(org_id, dataset_id, job_id, filename)
            storage.upload_bytes(
                client, settings.supabase_storage_bucket, result_storage_path, content, "text/csv"
            )

        db.update_job(
            client,
            job_id,
            org_id,
            status="succeeded",
            finished_at=_now(),
            result_summary=result_summary,
            result_storage_path=result_storage_path,
        )
        logger.info("job_succeeded", job_id=job_id, module=module, org_id=org_id)
    except Exception as exc:
        logger.error(
            "job_failed", job_id=job_id, module=module, org_id=org_id, error=str(exc), exc_info=True
        )
        db.update_job(
            client, job_id, org_id, status="failed", finished_at=_now(), error=_sanitize_error(exc)
        )


def _run_module(
    module: str, df: pd.DataFrame, params: dict[str, Any]
) -> tuple[dict[str, Any], tuple[str, bytes] | None]:
    if module == "forecast":
        result = forecast(df, ForecastParams(**params))
        summary = to_jsonable(dataclasses.asdict(result))
        return ForecastResultSchema(**summary).model_dump(mode="json"), None

    if module == "basket":
        result = analyze_baskets(df, BasketParams(**params))
        summary = to_jsonable(dataclasses.asdict(result))
        overflow_rules = summary.pop("overflow_rules")
        artifact = ("overflow_rules.csv", _rows_to_csv(overflow_rules)) if overflow_rules else None
        return BasketResultSchema(**summary).model_dump(mode="json"), artifact

    if module == "customer":
        result = model_customers(df, CustomerParams(**params))
        summary = to_jsonable(dataclasses.asdict(result))
        records = summary.pop("records")
        summary["records_preview"] = records[:RECORDS_PREVIEW_LIMIT]
        summary["records_truncated"] = len(records) > RECORDS_PREVIEW_LIMIT
        artifact = ("customers.csv", _rows_to_csv(records))
        return CustomerResultSchema(**summary).model_dump(mode="json"), artifact

    raise ValueError(f"Unknown module: {module!r}")


def _rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _sanitize_error(exc: Exception) -> str:
    """ValueErrors from `ml/` are actionable, user-facing validation messages
    (e.g. "not enough history"). Anything else is an unexpected internal
    error - don't leak details into a row the dataset owner can read.
    """
    if isinstance(exc, ValueError):
        return str(exc)
    return "An unexpected error occurred while processing this job."


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# ARQ task functions + worker settings
# ---------------------------------------------------------------------------


async def run_forecast_job(ctx: dict[str, Any], job_id: str) -> None:
    await process_job(get_service_client(), job_id)


async def run_basket_job(ctx: dict[str, Any], job_id: str) -> None:
    await process_job(get_service_client(), job_id)


async def run_customer_job(ctx: dict[str, Any], job_id: str) -> None:
    await process_job(get_service_client(), job_id)


async def startup(ctx: dict[str, Any]) -> None:
    configure_logging(get_settings().env)


class WorkerSettings:
    functions = [run_forecast_job, run_basket_job, run_customer_job]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
