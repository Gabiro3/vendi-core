"""Canonical column names and shared typed contracts for ML modules.

This is the single source of truth for:
- canonical transaction column names + accepted aliases
- the validation report shape
- input "params" dataclasses and output "result" dataclasses for each module

Hard rule: nothing in this package imports from `app`. Everything here is
plain dataclasses / stdlib so the `ml/` package stays framework-agnostic and
independently unit-testable. The API layer (Pydantic schemas) maps onto these
dataclasses at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Canonical transaction schema
# ---------------------------------------------------------------------------


class Col:
    """Canonical column names used throughout `ml/` after normalization."""

    TRANSACTION_ID = "transaction_id"
    CUSTOMER_ID = "customer_id"
    DATE = "date"
    PRODUCT_ID = "product_id"
    QUANTITY = "quantity"
    UNIT_PRICE = "unit_price"
    CATEGORY = "category"
    STORE_ID = "store_id"
    LINE_TOTAL = "line_total"  # derived: quantity * unit_price


REQUIRED_COLUMNS: tuple[str, ...] = (
    Col.TRANSACTION_ID,
    Col.DATE,
    Col.PRODUCT_ID,
    Col.QUANTITY,
    Col.UNIT_PRICE,
)

OPTIONAL_COLUMNS: tuple[str, ...] = (Col.CUSTOMER_ID, Col.CATEGORY, Col.STORE_ID)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Maps a normalized (lowercased, trimmed, non-alnum -> "_") header to its
# canonical column name. Validation normalizes incoming headers before
# looking them up here.
COLUMN_ALIASES: dict[str, str] = {
    "transaction_id": Col.TRANSACTION_ID,
    "order_id": Col.TRANSACTION_ID,
    "receipt_id": Col.TRANSACTION_ID,
    "receipt_no": Col.TRANSACTION_ID,
    "invoice_id": Col.TRANSACTION_ID,
    "invoice_no": Col.TRANSACTION_ID,
    "basket_id": Col.TRANSACTION_ID,
    "customer_id": Col.CUSTOMER_ID,
    "customerid": Col.CUSTOMER_ID,
    "client_id": Col.CUSTOMER_ID,
    "member_id": Col.CUSTOMER_ID,
    "loyalty_id": Col.CUSTOMER_ID,
    "date": Col.DATE,
    "order_date": Col.DATE,
    "transaction_date": Col.DATE,
    "invoice_date": Col.DATE,
    "purchase_date": Col.DATE,
    "timestamp": Col.DATE,
    "datetime": Col.DATE,
    "product_id": Col.PRODUCT_ID,
    "sku": Col.PRODUCT_ID,
    "item_id": Col.PRODUCT_ID,
    "itemid": Col.PRODUCT_ID,
    "stockcode": Col.PRODUCT_ID,
    "stock_code": Col.PRODUCT_ID,
    "quantity": Col.QUANTITY,
    "qty": Col.QUANTITY,
    "units": Col.QUANTITY,
    "unit_price": Col.UNIT_PRICE,
    "price": Col.UNIT_PRICE,
    "unitprice": Col.UNIT_PRICE,
    "sale_price": Col.UNIT_PRICE,
    "category": Col.CATEGORY,
    "product_category": Col.CATEGORY,
    "department": Col.CATEGORY,
    "category_name": Col.CATEGORY,
    "store_id": Col.STORE_ID,
    "store": Col.STORE_ID,
    "shop_id": Col.STORE_ID,
    "location_id": Col.STORE_ID,
    "branch_id": Col.STORE_ID,
}


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Persisted alongside a dataset row; summarizes ingest validation."""

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
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and self.rows_kept > 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["date_start"] = self.date_start.isoformat() if self.date_start else None
        d["date_end"] = self.date_end.isoformat() if self.date_end else None
        return d


# ---------------------------------------------------------------------------
# Forecasting (ml/forecasting)
# ---------------------------------------------------------------------------


@dataclass
class ForecastParams:
    horizon_days: int = 14
    granularity: str = "product"  # "product" | "product_store"
    quantiles: tuple[float, ...] = (0.5, 0.9)
    country_code: str = "US"
    # Series with fewer than this many days of history fall back to a
    # category-level model / seasonal-naive baseline and are flagged.
    min_history_days: int = 28
    # Lead time (days) used to size `recommended_order_qty` from the
    # upper-quantile forecast.
    lead_time_days: int = 7
    # Number of trailing `horizon_days`-length windows held out for backtest.
    backtest_windows: int = 2


@dataclass
class ForecastPoint:
    date: date
    forecast: float
    lower: float
    upper: float


@dataclass
class SeriesForecast:
    series_id: str
    product_id: str
    store_id: str | None
    category: str | None
    points: list[ForecastPoint]
    trend_pct: float | None
    recommended_order_qty: float
    low_confidence: bool


@dataclass
class BacktestMetrics:
    mase: float | None
    rmse: float | None
    baseline_mase: float | None
    baseline_rmse: float | None
    beats_baseline: bool | None
    n_windows: int


@dataclass
class ForecastResult:
    series: list[SeriesForecast]
    backtest: BacktestMetrics
    feature_importances: dict[str, float]
    horizon_days: int
    granularity: str
    quantiles: tuple[float, ...]
    n_series: int
    n_low_confidence: int
    generated_at: datetime
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market-basket analysis (ml/basket)
# ---------------------------------------------------------------------------


@dataclass
class BasketParams:
    level: str = "product"  # "product" | "category"
    dimension: str | None = None  # None | "store" | "day_of_week" | "time_of_day"
    min_support: float = 0.005
    min_confidence: float = 0.2
    min_lift: float = 1.2
    max_len: int = 2
    max_rules: int = 200


@dataclass
class BasketRule:
    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float
    decision: str
    slice_key: str | None = None


@dataclass
class SliceCoverage:
    slice_key: str | None
    n_transactions: int
    n_itemsets: int
    n_rules_before_cap: int


@dataclass
class BasketResult:
    rules: list[BasketRule]
    total_rules_found: int
    rules_truncated: bool
    coverage: list[SliceCoverage]
    level: str
    dimension: str | None
    warnings: list[str] = field(default_factory=list)
    # Rules beyond `params.max_rules` (by lift). Not part of result_summary;
    # the worker writes these to a CSV artifact when non-empty.
    overflow_rules: list[BasketRule] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Customer modeling (ml/customer)
# ---------------------------------------------------------------------------


@dataclass
class CustomerParams:
    forecast_period_days: int = 365
    discount_rate: float = 0.01
    n_value_segments: int = 5
    # If True, also emit the iso-value contour grid (frequency x recency).
    include_iso_value_grid: bool = True


@dataclass
class CustomerRecord:
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


@dataclass
class SegmentSummary:
    segment: str
    count: int
    avg_clv: float
    pct_of_total_value: float


@dataclass
class IsoValueGridPoint:
    frequency: float
    recency: float
    clv: float


@dataclass
class CustomerResult:
    records: list[CustomerRecord]
    segment_summary: list[SegmentSummary]
    gamma_gamma_corr: float
    gamma_gamma_assumption_ok: bool
    bgnbd_params: dict[str, float]
    gamma_gamma_params: dict[str, float] | None
    iso_value_grid: list[IsoValueGridPoint] | None
    observation_period_days: int
    forecast_period_days: int
    n_customers: int
    n_repeat_customers: int
    warnings: list[str] = field(default_factory=list)
