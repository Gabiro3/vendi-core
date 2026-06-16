"""Schemas for `/forecast/*` endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel


class ForecastRunParams(APIModel):
    horizon_days: int = Field(default=14, ge=1, le=90)
    granularity: Literal["product", "product_store"] = "product"
    quantiles: list[float] = Field(default_factory=lambda: [0.5, 0.9])
    country_code: str = "US"


class ForecastRunRequest(APIModel):
    dataset_id: str
    params: ForecastRunParams = Field(default_factory=ForecastRunParams)
    force: bool = False


class ForecastPointSchema(APIModel):
    date: date
    forecast: float
    lower: float
    upper: float


class SeriesForecastSchema(APIModel):
    series_id: str
    product_id: str
    store_id: str | None
    category: str | None
    points: list[ForecastPointSchema]
    trend_pct: float | None
    recommended_order_qty: float
    low_confidence: bool


class BacktestMetricsSchema(APIModel):
    mase: float | None
    rmse: float | None
    baseline_mase: float | None
    baseline_rmse: float | None
    beats_baseline: bool | None
    n_windows: int


class ForecastResultSchema(APIModel):
    series: list[SeriesForecastSchema]
    backtest: BacktestMetricsSchema
    feature_importances: dict[str, float]
    horizon_days: int
    granularity: str
    quantiles: list[float]
    n_series: int
    n_low_confidence: int
    generated_at: datetime
    warnings: list[str]
