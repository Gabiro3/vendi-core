"""Transaction schema validation.

Pure pandas. Normalizes raw column headers to the canonical schema in
`ml.contracts`, coerces types, drops invalid rows with a reason, deduplicates,
and produces a `ValidationReport`. Fails loudly (via `report.errors`) when the
data cannot support the canonical schema at all; otherwise it cleans what it
can and reports exactly what was dropped and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from ml.contracts import ALL_COLUMNS, COLUMN_ALIASES, REQUIRED_COLUMNS, Col, ValidationReport

_HEADER_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ValidationConfig:
    """Thresholds are configurable; values below them produce warnings, not errors."""

    min_transactions: int = 200
    min_date_span_weeks: int = 8
    max_rows: int = 5_000_000
    max_columns: int = 200
    # Rows with a date more than this many days in the future are dropped.
    future_date_tolerance_days: int = 1
    today: date | None = None  # injectable for tests; defaults to UTC today


def normalize_header(name: str) -> str:
    """Lowercase, trim, and collapse non-alphanumerics to underscores."""
    normalized = _HEADER_NORMALIZE_RE.sub("_", str(name).strip().lower()).strip("_")
    return normalized


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns to canonical names where recognized.

    Returns the renamed dataframe and a mapping of {canonical_name: original_header}
    for columns that were successfully mapped.
    """
    rename_map: dict[str, str] = {}
    mapped: dict[str, str] = {}
    for original in df.columns:
        normalized = normalize_header(original)
        canonical = COLUMN_ALIASES.get(normalized, normalized if normalized in ALL_COLUMNS else None)
        if canonical is None:
            continue
        if canonical in mapped:
            # First match wins; later duplicate-mapped columns are left unmapped
            # (and therefore dropped) rather than silently overwriting data.
            continue
        rename_map[original] = canonical
        mapped[canonical] = original

    out = df.rename(columns=rename_map)
    # Drop any columns that didn't map to a canonical name and aren't already canonical.
    keep = [c for c in out.columns if c in ALL_COLUMNS]
    out = out.loc[:, keep]
    return out, mapped


def validate_transactions(
    df: pd.DataFrame, config: ValidationConfig | None = None
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a raw transactions dataframe.

    Returns (clean_df, report). If `report.is_valid` is False, `clean_df` may
    be empty or unusable and the caller (API layer) should fail the dataset
    ingest with `report.errors`.
    """
    config = config or ValidationConfig()
    today = config.today or datetime.now(UTC).date()

    rows_in = len(df)
    errors: list[str] = []
    warnings: list[str] = []
    dropped_by_reason: dict[str, int] = {}

    if rows_in > config.max_rows:
        errors.append(
            f"Too many rows ({rows_in}); max allowed is {config.max_rows}."
        )
    if df.shape[1] > config.max_columns:
        errors.append(
            f"Too many columns ({df.shape[1]}); max allowed is {config.max_columns}."
        )

    df, mapped = normalize_columns(df)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        errors.append(
            "Missing required column(s): " + ", ".join(missing_required)
            + ". Recognized columns: " + ", ".join(sorted(mapped.values()))
        )

    if errors:
        return pd.DataFrame(columns=list(ALL_COLUMNS)), ValidationReport(
            rows_in=rows_in,
            rows_kept=0,
            rows_dropped=rows_in,
            dropped_by_reason=dropped_by_reason,
            duplicate_rows_removed=0,
            date_start=None,
            date_end=None,
            distinct_transactions=0,
            distinct_products=0,
            distinct_customers=None,
            distinct_stores=None,
            has_customer_id=False,
            has_category=False,
            has_store_id=False,
            warnings=warnings,
            errors=errors,
        )

    out = df.copy()

    def _drop_mask(mask: pd.Series, reason: str) -> None:
        """Remove rows where mask is True, tallying the count under `reason`."""
        nonlocal out
        n = int(mask.sum())
        if n:
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + n
            out = out.loc[~mask]

    # --- Required string identifiers: not null ---
    for col, reason in (
        (Col.TRANSACTION_ID, "missing_transaction_id"),
        (Col.PRODUCT_ID, "missing_product_id"),
    ):
        out[col] = out[col].astype("string")
        _drop_mask(out[col].isna() | (out[col].str.strip() == ""), reason)

    # --- Date: parse, drop null/unparseable, drop future-dated ---
    parsed_date = pd.to_datetime(out[Col.DATE], errors="coerce", utc=True)
    _drop_mask(parsed_date.isna(), "invalid_date")
    parsed_date = parsed_date.loc[out.index]
    future_cutoff = pd.Timestamp(today, tz="UTC") + pd.Timedelta(
        days=config.future_date_tolerance_days
    )
    _drop_mask(parsed_date > future_cutoff, "future_date")
    out[Col.DATE] = parsed_date.loc[out.index].dt.tz_localize(None).dt.normalize()

    # --- Quantity: numeric, > 0 ---
    qty = pd.to_numeric(out[Col.QUANTITY], errors="coerce")
    _drop_mask(qty.isna(), "invalid_quantity")
    qty = qty.loc[out.index]
    _drop_mask(qty <= 0, "non_positive_quantity")
    out[Col.QUANTITY] = qty.loc[out.index].astype(float)

    # --- Unit price: numeric, >= 0 ---
    price = pd.to_numeric(out[Col.UNIT_PRICE], errors="coerce")
    _drop_mask(price.isna(), "invalid_unit_price")
    price = price.loc[out.index]
    _drop_mask(price < 0, "negative_unit_price")
    out[Col.UNIT_PRICE] = price.loc[out.index].astype(float)

    # --- Optional columns: normalize types, blank -> NA ---
    has_customer_id = Col.CUSTOMER_ID in out.columns
    if has_customer_id:
        out[Col.CUSTOMER_ID] = out[Col.CUSTOMER_ID].astype("string")
        out.loc[out[Col.CUSTOMER_ID].str.strip() == "", Col.CUSTOMER_ID] = pd.NA

    has_category = Col.CATEGORY in out.columns
    if has_category:
        out[Col.CATEGORY] = out[Col.CATEGORY].astype("string").str.strip()
        out.loc[out[Col.CATEGORY] == "", Col.CATEGORY] = pd.NA

    has_store_id = Col.STORE_ID in out.columns
    if has_store_id:
        out[Col.STORE_ID] = out[Col.STORE_ID].astype("string")
        out.loc[out[Col.STORE_ID].str.strip() == "", Col.STORE_ID] = pd.NA

    # --- Derived ---
    out[Col.LINE_TOTAL] = out[Col.QUANTITY] * out[Col.UNIT_PRICE]

    # --- Exact-duplicate rows ---
    before = len(out)
    out = out.drop_duplicates()
    duplicate_rows_removed = before - len(out)

    rows_kept = len(out)
    rows_dropped = rows_in - rows_kept

    # --- Summary stats ---
    if rows_kept:
        date_start = out[Col.DATE].min().date()
        date_end = out[Col.DATE].max().date()
        date_span_days = (date_end - date_start).days
    else:
        date_start = None
        date_end = None
        date_span_days = 0

    distinct_transactions = int(out[Col.TRANSACTION_ID].nunique()) if rows_kept else 0
    distinct_products = int(out[Col.PRODUCT_ID].nunique()) if rows_kept else 0
    distinct_customers = (
        int(out[Col.CUSTOMER_ID].dropna().nunique()) if has_customer_id and rows_kept else None
    )
    distinct_stores = (
        int(out[Col.STORE_ID].dropna().nunique()) if has_store_id and rows_kept else None
    )

    # --- Threshold warnings (non-fatal) ---
    if rows_kept == 0:
        errors.append("No valid rows remained after validation.")
    else:
        if distinct_transactions < config.min_transactions:
            warnings.append(
                f"Only {distinct_transactions} distinct transactions found; "
                f"recommend at least {config.min_transactions} for reliable "
                "forecasting/CLV results."
            )
        if date_span_days < config.min_date_span_weeks * 7:
            warnings.append(
                f"Date range spans only {date_span_days} days; recommend at "
                f"least {config.min_date_span_weeks * 7} days "
                f"({config.min_date_span_weeks} weeks) for reliable results."
            )
        if not has_customer_id:
            warnings.append(
                "No customer_id column found; the customer module cannot run "
                "on this dataset."
            )

    out = out.reset_index(drop=True)

    report = ValidationReport(
        rows_in=rows_in,
        rows_kept=rows_kept,
        rows_dropped=rows_dropped,
        dropped_by_reason=dropped_by_reason,
        duplicate_rows_removed=duplicate_rows_removed,
        date_start=date_start,
        date_end=date_end,
        distinct_transactions=distinct_transactions,
        distinct_products=distinct_products,
        distinct_customers=distinct_customers,
        distinct_stores=distinct_stores,
        has_customer_id=has_customer_id,
        has_category=has_category,
        has_store_id=has_store_id,
        warnings=warnings,
        errors=errors,
    )
    return out, report
