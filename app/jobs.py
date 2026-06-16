"""ARQ enqueue helpers.

The API never runs ML modules in-process (except under `SYNC_JOBS=true`,
which is for local smoke tests only — see TECHNICAL_SPEC.md §2). It creates a
`jobs` row in `queued` state, then enqueues an ARQ task carrying the `job_id`.
The worker (`app/worker.py`) looks the job up, loads the dataset, and runs the
corresponding `ml/` module.
"""

from __future__ import annotations

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.config import Settings

MODULE_TASK_NAMES = {
    "forecast": "run_forecast_job",
    "basket": "run_basket_job",
    "customer": "run_customer_job",
}


async def get_redis_pool(settings: Settings) -> ArqRedis:
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def enqueue_module_job(redis_pool: ArqRedis, job_id: str, module: str) -> None:
    task_name = MODULE_TASK_NAMES.get(module)
    if task_name is None:
        raise ValueError(f"Unknown module: {module!r}")
    await redis_pool.enqueue_job(task_name, job_id)
