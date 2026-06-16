"""Shared feature engineering for the forecasting module.

All functions are pure: dataframe in, dataframe out. Used by both
`ml/forecasting/model.py` (training/inference) and `ml/forecasting/evaluate.py`
(backtesting), so feature definitions can never drift between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from holidays import country_holidays

from ml.contracts import Col

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 28)


def build_forecast_view(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate cleaned transactions to a daily series per (product[, store]).

    Output columns: date, product_id[, store_id], category, quantity,
    unit_price (quantity-weighted average for the day), with every series
    reindexed to a continuous daily date range and missing days filled with
    quantity=0 (retail demand is intermittent: absence of sales is signal).

    `granularity`: "product" or "product_store".
    """
    if granularity not in ("product", "product_store"):
        raise ValueError(f"Unknown granularity: {granularity!r}")

    group_cols = [Col.PRODUCT_ID, Col.STORE_ID] if granularity == "product_store" else [
        Col.PRODUCT_ID
    ]
    if granularity == "product_store" and Col.STORE_ID not in df.columns:
        raise ValueError("granularity='product_store' requires a store_id column")

    work = df.copy()
    work[Col.DATE] = pd.to_datetime(work[Col.DATE])

    daily = (
        work.groupby(group_cols + [Col.DATE], as_index=False)
        .agg(
            **{
                Col.QUANTITY: (Col.QUANTITY, "sum"),
                Col.LINE_TOTAL: (Col.LINE_TOTAL, "sum"),
            }
        )
    )
    daily[Col.UNIT_PRICE] = np.where(
        daily[Col.QUANTITY] > 0, daily[Col.LINE_TOTAL] / daily[Col.QUANTITY], np.nan
    )

    # Most recent category per product (categories rarely change; last wins).
    category_map = None
    if Col.CATEGORY in work.columns:
        category_map = (
            work.sort_values(Col.DATE)
            .groupby(Col.PRODUCT_ID)[Col.CATEGORY]
            .last()
        )

    date_range = pd.date_range(work[Col.DATE].min(), work[Col.DATE].max(), freq="D")

    series_frames = []
    for key, group in daily.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        full = (
            group.set_index(Col.DATE)
            .reindex(date_range)
            .rename_axis(Col.DATE)
        )
        full[Col.QUANTITY] = full[Col.QUANTITY].fillna(0.0)
        full[Col.UNIT_PRICE] = full[Col.UNIT_PRICE].ffill().bfill()
        for col, val in zip(group_cols, keys, strict=True):
            full[col] = val
        full = full.drop(columns=[Col.LINE_TOTAL])
        full = full.reset_index()
        series_frames.append(full)

    out = pd.concat(series_frames, ignore_index=True)

    if category_map is not None:
        out[Col.CATEGORY] = out[Col.PRODUCT_ID].map(category_map)
    else:
        out[Col.CATEGORY] = pd.NA

    # If price was never observed for a series (shouldn't happen post-fill,
    # but guards an all-zero series), fall back to the global median.
    out[Col.UNIT_PRICE] = out[Col.UNIT_PRICE].fillna(out[Col.UNIT_PRICE].median())

    return out.sort_values(group_cols + [Col.DATE]).reset_index(drop=True)


def add_calendar_features(
    df: pd.DataFrame, date_col: str = Col.DATE, country_code: str = "US"
) -> pd.DataFrame:
    """Add day-of-week, seasonality, payday-proximity, and holiday features."""
    out = df.copy()
    dates = pd.to_datetime(out[date_col])

    out["day_of_week"] = dates.dt.dayofweek.astype("int16")
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    out["month"] = dates.dt.month.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = dates.dt.is_month_start.astype("int8")
    out["is_month_end"] = dates.dt.is_month_end.astype("int8")

    # Payday proximity: many retail purchases cluster around the 1st, 15th,
    # and last day of the month. Encode as min distance (days) to the nearest
    # of those anchors.
    day = dates.dt.day
    days_in_month = dates.dt.days_in_month
    dist_to_1 = day.astype("int16")
    dist_to_15 = (day - 15).abs().astype("int16")
    dist_to_end = (days_in_month - day).astype("int16")
    out["payday_proximity"] = np.minimum(np.minimum(dist_to_1, dist_to_15), dist_to_end).astype(
        "int16"
    )

    years = sorted({d.year for d in dates})
    try:
        holiday_dates = set(country_holidays(country_code, years=years).keys())
    except (NotImplementedError, KeyError):
        holiday_dates = set()
    out["is_holiday"] = dates.dt.date.isin(holiday_dates).astype("int8")

    return out


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = Col.QUANTITY,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add `<target>_lag_<n>` columns, computed within each series.

    Assumes `df` is sorted by `group_cols + [date]`.
    """
    out = df.copy()
    grouped = out.groupby(group_cols)[target_col]
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = Col.QUANTITY,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add `<target>_rollmean_<w>` / `<target>_rollstd_<w>`, shifted by 1 day
    so the window for date `t` only sees data through `t-1` (no leakage).
    """
    out = df.copy()
    shifted = out.groupby(group_cols)[target_col].shift(1)
    out["_shifted_for_rolling"] = shifted
    for window in windows:
        rolled = out.groupby(group_cols)["_shifted_for_rolling"]
        out[f"{target_col}_rollmean_{window}"] = rolled.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        out[f"{target_col}_rollstd_{window}"] = rolled.transform(
            lambda s, w=window: s.rolling(w, min_periods=2).std()
        )
    out = out.drop(columns=["_shifted_for_rolling"])
    return out


def add_price_features(
    df: pd.DataFrame,
    group_cols: list[str],
    price_col: str = Col.UNIT_PRICE,
    window: int = 28,
) -> pd.DataFrame:
    """Add a trailing median price (shifted) and a discount-vs-trailing-median ratio.

    `promo_flag` = 1 when the current price is materially below the trailing
    median (a simple proxy when no explicit promo flag is supplied).
    """
    out = df.copy()
    shifted_price = out.groupby(group_cols)[price_col].shift(1)
    out["_shifted_price"] = shifted_price
    out["price_trailing_median"] = out.groupby(group_cols)["_shifted_price"].transform(
        lambda s: s.rolling(window, min_periods=1).median()
    )
    out["price_trailing_median"] = out["price_trailing_median"].fillna(out[price_col])
    out["discount_pct"] = np.where(
        out["price_trailing_median"] > 0,
        1.0 - (out[price_col] / out["price_trailing_median"]),
        0.0,
    )
    out["promo_flag"] = (out["discount_pct"] > 0.05).astype("int8")
    out = out.drop(columns=["_shifted_price"])
    return out


def build_model_features(
    df: pd.DataFrame,
    group_cols: list[str],
    country_code: str = "US",
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Apply the full shared feature pipeline in the canonical order.

    `df` must already be the output of `build_forecast_view`, sorted by
    `group_cols + [date]`.
    """
    out = add_calendar_features(df, country_code=country_code)
    out = add_lag_features(out, group_cols, lags=lags)
    out = add_rolling_features(out, group_cols, windows=rolling_windows)
    out = add_price_features(out, group_cols)
    return out


def feature_columns(
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> list[str]:
    """The full list of model feature columns produced by `build_model_features`,
    excluding categorical IDs (caller appends those based on granularity).
    """
    cols = [
        "day_of_week",
        "week_of_year",
        "month",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "payday_proximity",
        "is_holiday",
        Col.UNIT_PRICE,
        "price_trailing_median",
        "discount_pct",
        "promo_flag",
    ]
    cols += [f"{Col.QUANTITY}_lag_{lag}" for lag in lags]
    for window in rolling_windows:
        cols.append(f"{Col.QUANTITY}_rollmean_{window}")
        cols.append(f"{Col.QUANTITY}_rollstd_{window}")
    return cols
