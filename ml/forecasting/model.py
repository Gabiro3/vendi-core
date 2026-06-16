"""Demand forecasting: pooled LightGBM (tweedie point forecast + quantile
prediction intervals) over per-product or per-product-store daily series,
with recursive multi-step forecasting and a seasonal-naive cold-start
fallback for series with too little history.

`forecast(df, params) -> ForecastResult` is the sole entrypoint.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from holidays import country_holidays

from ml.contracts import (
    Col,
    ForecastParams,
    ForecastPoint,
    ForecastResult,
    SeriesForecast,
)
from ml.features import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    build_forecast_view,
    build_model_features,
    feature_columns,
)
from ml.forecasting.evaluate import run_backtest, train_lgbm_model


def forecast(df: pd.DataFrame, params: ForecastParams) -> ForecastResult:
    warnings: list[str] = []
    quantiles = sorted({float(q) for q in params.quantiles}) or [0.5]

    group_cols = (
        [Col.PRODUCT_ID, Col.STORE_ID] if params.granularity == "product_store" else [Col.PRODUCT_ID]
    )

    view = build_forecast_view(df, params.granularity)
    categorical_cols = [c for c in (Col.PRODUCT_ID, Col.CATEGORY, Col.STORE_ID) if c in view.columns]

    featured = build_model_features(view, group_cols, country_code=params.country_code)
    for col in categorical_cols:
        featured[col] = featured[col].astype("category")

    feat_cols = feature_columns()
    model_feature_cols = feat_cols + categorical_cols

    max_lag = max(DEFAULT_LAGS)
    train_df = featured.dropna(subset=[f"{Col.QUANTITY}_lag_{max_lag}"])
    if train_df.empty:
        raise ValueError(
            "Not enough history to train a forecasting model "
            f"(need more than {max_lag} days of history per series)."
        )

    X_train = train_df[model_feature_cols]
    y_train = train_df[Col.QUANTITY]

    tweedie_model = train_lgbm_model(X_train, y_train, categorical_cols, objective="tweedie")
    quantile_models = {
        q: train_lgbm_model(X_train, y_train, categorical_cols, objective="quantile", alpha=q)
        for q in quantiles
    }

    feature_importances = dict(
        sorted(
            zip(model_feature_cols, (float(v) for v in tweedie_model.feature_importances_), strict=False),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    backtest_metrics, backtest_warnings = run_backtest(
        featured, categorical_cols, model_feature_cols, params
    )
    warnings.extend(backtest_warnings)

    history_lengths = _history_lengths(view, group_cols)
    holiday_dates = _holiday_dates(view, params)

    series_results: list[SeriesForecast] = []
    n_low_confidence = 0

    for key, group in featured.groupby(group_cols, sort=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group = group.sort_values(Col.DATE).reset_index(drop=True)
        history_len = history_lengths.get(key_tuple, 0)
        low_confidence = history_len < params.min_history_days

        if low_confidence:
            n_low_confidence += 1
            points = _seasonal_naive_points(group, params.horizon_days)
        else:
            categorical_values = {c: group[c].iloc[-1] for c in categorical_cols}
            points = _recursive_forecast_series(
                group,
                tweedie_model,
                quantile_models,
                model_feature_cols,
                categorical_values,
                params,
                holiday_dates,
            )

        product_id = str(group[Col.PRODUCT_ID].iloc[-1])
        store_id = str(group[Col.STORE_ID].iloc[-1]) if Col.STORE_ID in group.columns else None
        category = (
            str(group[Col.CATEGORY].iloc[-1])
            if Col.CATEGORY in group.columns and pd.notna(group[Col.CATEGORY].iloc[-1])
            else None
        )

        series_id = "|".join(str(k) for k in key_tuple)
        forecast_mean = float(np.mean([p.forecast for p in points])) if points else 0.0

        lookback = min(params.horizon_days, len(group))
        recent_actual_mean = float(group[Col.QUANTITY].tail(lookback).mean()) if lookback else 0.0
        trend_pct = (
            (forecast_mean - recent_actual_mean) / recent_actual_mean * 100.0
            if recent_actual_mean > 0
            else None
        )

        lead_time = min(params.lead_time_days, len(points))
        recommended_order_qty = float(sum(p.upper for p in points[:lead_time]))

        series_results.append(
            SeriesForecast(
                series_id=series_id,
                product_id=product_id,
                store_id=store_id,
                category=category,
                points=points,
                trend_pct=trend_pct,
                recommended_order_qty=recommended_order_qty,
                low_confidence=low_confidence,
            )
        )

    return ForecastResult(
        series=series_results,
        backtest=backtest_metrics,
        feature_importances=feature_importances,
        horizon_days=params.horizon_days,
        granularity=params.granularity,
        quantiles=tuple(quantiles),
        n_series=len(series_results),
        n_low_confidence=n_low_confidence,
        generated_at=datetime.now(UTC),
        warnings=warnings,
    )


def _history_lengths(view: pd.DataFrame, group_cols: list[str]) -> dict[tuple, int]:
    """Days from a series' first non-zero sale to the last date in the
    dataset. Series shorter than `params.min_history_days` are treated as
    cold-start (newly listed products) and use the seasonal-naive fallback.
    """
    lengths: dict[tuple, int] = {}
    for key, group in view.groupby(group_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        nonzero = group[group[Col.QUANTITY] > 0]
        if nonzero.empty:
            lengths[key_tuple] = 0
        else:
            first = nonzero[Col.DATE].min()
            last = group[Col.DATE].max()
            lengths[key_tuple] = int((last - first).days) + 1
    return lengths


def _holiday_dates(view: pd.DataFrame, params: ForecastParams) -> set:
    last_date = view[Col.DATE].max()
    horizon_end = last_date + pd.Timedelta(days=params.horizon_days)
    years = sorted({d.year for d in (view[Col.DATE].min(), last_date, horizon_end)})
    try:
        return set(country_holidays(params.country_code, years=years).keys())
    except (NotImplementedError, KeyError):
        return set()


def _seasonal_naive_points(history: pd.DataFrame, horizon_days: int) -> list[ForecastPoint]:
    """Cold-start fallback: per-day-of-week mean (with overall mean as a
    backstop), flagged `low_confidence` by the caller.
    """
    last_date = history[Col.DATE].iloc[-1]
    by_dow = history.groupby(history[Col.DATE].dt.dayofweek)[Col.QUANTITY].agg(["mean", "std"])
    overall_mean = float(history[Col.QUANTITY].mean())
    overall_std = float(history[Col.QUANTITY].std(ddof=1)) if len(history) >= 2 else 0.0
    overall_std = 0.0 if np.isnan(overall_std) else overall_std

    points = []
    for step in range(1, horizon_days + 1):
        new_date = last_date + pd.Timedelta(days=step)
        dow = new_date.dayofweek
        if dow in by_dow.index and not np.isnan(by_dow.loc[dow, "mean"]):
            mean = float(by_dow.loc[dow, "mean"])
            std = by_dow.loc[dow, "std"]
            std = float(std) if not np.isnan(std) else overall_std
        else:
            mean, std = overall_mean, overall_std
        mean = max(mean, 0.0)
        std = max(std, 0.0)
        points.append(
            ForecastPoint(
                date=new_date.date(),
                forecast=mean,
                lower=max(mean - std, 0.0),
                upper=mean + std,
            )
        )
    return points


def _recursive_forecast_series(
    history: pd.DataFrame,
    tweedie_model,
    quantile_models: dict[float, object],
    model_feature_cols: list[str],
    categorical_values: dict[str, object],
    params: ForecastParams,
    holiday_dates: set,
) -> list[ForecastPoint]:
    """Step the model forward one day at a time, feeding each step's point
    forecast back into the lag/rolling features for the next step (since
    future actuals don't exist yet).
    """
    quantities: list[float] = history[Col.QUANTITY].tolist()
    last_date = history[Col.DATE].iloc[-1]
    last_price = float(history[Col.UNIT_PRICE].iloc[-1])
    price_trailing_median = float(history["price_trailing_median"].iloc[-1])

    points: list[ForecastPoint] = []
    for step in range(1, params.horizon_days + 1):
        new_date = last_date + pd.Timedelta(days=step)
        row: dict[str, object] = {
            "day_of_week": new_date.dayofweek,
            "week_of_year": int(new_date.isocalendar().week),
            "month": new_date.month,
            "is_weekend": int(new_date.dayofweek >= 5),
            "is_month_start": int(new_date.is_month_start),
            "is_month_end": int(new_date.is_month_end),
            "is_holiday": int(new_date.date() in holiday_dates),
            Col.UNIT_PRICE: last_price,
            "price_trailing_median": price_trailing_median,
            "discount_pct": 0.0,
            "promo_flag": 0,
        }
        day = new_date.day
        days_in_month = new_date.days_in_month
        row["payday_proximity"] = min(day, abs(day - 15), days_in_month - day)

        for lag in DEFAULT_LAGS:
            row[f"{Col.QUANTITY}_lag_{lag}"] = quantities[-lag] if len(quantities) >= lag else np.nan

        for window in DEFAULT_ROLLING_WINDOWS:
            window_vals = quantities[-window:]
            row[f"{Col.QUANTITY}_rollmean_{window}"] = float(np.mean(window_vals))
            row[f"{Col.QUANTITY}_rollstd_{window}"] = (
                float(np.std(window_vals, ddof=1)) if len(window_vals) >= 2 else np.nan
            )

        row.update(categorical_values)

        X_row = pd.DataFrame([row])[model_feature_cols]
        for col in categorical_values:
            X_row[col] = X_row[col].astype("category")

        point = max(float(tweedie_model.predict(X_row)[0]), 0.0)
        if quantile_models:
            q_preds = [max(float(m.predict(X_row)[0]), 0.0) for m in quantile_models.values()]
            lower, upper = min(q_preds + [point]), max(q_preds + [point])
        else:
            lower, upper = point, point

        points.append(ForecastPoint(date=new_date.date(), forecast=point, lower=lower, upper=upper))
        quantities.append(point)

    return points
