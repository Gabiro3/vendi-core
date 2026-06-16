"""Application settings, env-driven via pydantic-settings.

All secrets and environment-specific values live here. Never hardcode
credentials or URLs elsewhere in `app/`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Supabase ---
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_jwks_url: str = ""
    supabase_storage_bucket: str = "vendi-datasets"

    # --- Job queue ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Upload / validation limits ---
    max_upload_mb: int = 100
    min_date_span_weeks: int = 8
    min_transactions: int = 200

    # --- CORS ---
    allowed_origins: str = "http://localhost:3000"

    # --- Environment ---
    env: str = "dev"

    # Run module jobs synchronously in-process instead of via ARQ.
    # For local smoke tests only - never enable in prod.
    sync_jobs: bool = False

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
