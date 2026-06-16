"""Schemas for `/customer/*` endpoints."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import APIModel

# Inline preview size for the per-customer table. The full table (which can
# be tens of thousands of rows) is written to a CSV artifact in Storage and
# fetched via `/jobs/{id}/artifact`.
RECORDS_PREVIEW_LIMIT = 100


class CustomerRunParams(APIModel):
    forecast_period_days: int = Field(default=365, ge=30, le=1095)
    discount_rate: float = Field(default=0.01, ge=0, le=1)
    n_value_segments: int = Field(default=5, ge=2, le=10)
    include_iso_value_grid: bool = True


class CustomerRunRequest(APIModel):
    dataset_id: str
    params: CustomerRunParams = Field(default_factory=CustomerRunParams)
    force: bool = False


class CustomerRecordSchema(APIModel):
    customer_id: str
    frequency: float
    recency: float
    T: float
    monetary_value: float
    predicted_purchases: float
    prob_alive: float
    predicted_avg_value: float
    clv: float
    value_segment: str
    action: str
    is_at_risk_high_value: bool


class SegmentSummarySchema(APIModel):
    segment: str
    count: int
    avg_clv: float
    pct_of_total_value: float


class IsoValueGridPointSchema(APIModel):
    frequency: float
    recency: float
    clv: float


class CustomerResultSchema(APIModel):
    records_preview: list[CustomerRecordSchema]
    records_truncated: bool
    segment_summary: list[SegmentSummarySchema]
    gamma_gamma_corr: float
    gamma_gamma_assumption_ok: bool
    bgnbd_params: dict[str, float]
    gamma_gamma_params: dict[str, float] | None
    iso_value_grid: list[IsoValueGridPointSchema] | None
    observation_period_days: int
    forecast_period_days: int
    n_customers: int
    n_repeat_customers: int
    warnings: list[str]
