"""Schemas for `/basket/*` endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel


class BasketRunParams(APIModel):
    level: Literal["product", "category"] = "product"
    dimension: Literal["store", "day_of_week", "time_of_day"] | None = None
    min_support: float = Field(default=0.005, gt=0, le=1)
    min_confidence: float = Field(default=0.2, ge=0, le=1)
    min_lift: float = Field(default=1.2, ge=0)
    max_len: int = Field(default=2, ge=2, le=3)


class BasketRunRequest(APIModel):
    dataset_id: str
    params: BasketRunParams = Field(default_factory=BasketRunParams)
    force: bool = False


class BasketRuleSchema(APIModel):
    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float
    decision: str
    slice_key: str | None


class SliceCoverageSchema(APIModel):
    slice_key: str | None
    n_transactions: int
    n_itemsets: int
    n_rules_before_cap: int


class BasketResultSchema(APIModel):
    rules: list[BasketRuleSchema]
    total_rules_found: int
    rules_truncated: bool
    coverage: list[SliceCoverageSchema]
    level: str
    dimension: str | None
    warnings: list[str]
