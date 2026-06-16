"""FastAPI app factory: middleware, exception handlers, routers, and the ARQ
redis pool lifecycle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import jobs as job_queue
from app.config import get_settings
from app.errors import register_exception_handlers
from app.logging import configure_logging, install_request_id_middleware
from app.routers import basket, customer, dashboard, datasets, forecast, jobs, me


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.redis_pool = None if settings.sync_jobs else await job_queue.get_redis_pool(settings)
    try:
        yield
    finally:
        if app.state.redis_pool is not None:
            await app.state.redis_pool.close()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.env)

    app = FastAPI(title="Vendi Retail Intelligence API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_request_id_middleware(app)
    register_exception_handlers(app)

    app.include_router(me.router)
    app.include_router(dashboard.router)
    app.include_router(datasets.router)
    app.include_router(forecast.router)
    app.include_router(basket.router)
    app.include_router(customer.router)
    app.include_router(jobs.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
