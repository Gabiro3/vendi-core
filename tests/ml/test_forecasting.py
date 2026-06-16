from __future__ import annotations

import pandas as pd
import pytest

from ml.contracts import Col, ForecastParams
from ml.forecasting.model import forecast


@pytest.fixture(scope="module")
def forecast_result(forecast_transactions: pd.DataFrame):
    params = ForecastParams(horizon_days=14, granularity="product", quantiles=(0.5, 0.9))
    return forecast(forecast_transactions, params)


def test_forecast_runs_end_to_end_with_expected_shape(forecast_result):
    result = forecast_result

    assert result.n_series == 4  # SKU-A, SKU-B, SKU-C, SKU-NEW
    assert result.horizon_days == 14
    assert result.granularity == "product"
    assert result.quantiles == (0.5, 0.9)

    for series in result.series:
        assert series.product_id is not None
        assert len(series.points) == 14
        for point in series.points:
            assert point.lower <= point.forecast <= point.upper
            assert point.forecast >= 0
        assert series.recommended_order_qty >= 0


def test_cold_start_product_is_low_confidence(forecast_result):
    result = forecast_result
    by_product = {s.product_id: s for s in result.series}

    assert by_product["SKU-NEW"].low_confidence is True
    assert result.n_low_confidence >= 1

    for sku in ("SKU-A", "SKU-B", "SKU-C"):
        assert by_product[sku].low_confidence is False


def test_backtest_metrics_are_populated(forecast_result):
    backtest = forecast_result.backtest

    assert backtest.n_windows > 0
    assert backtest.rmse is not None
    assert backtest.baseline_rmse is not None
    assert backtest.rmse >= 0
    assert backtest.baseline_rmse >= 0


def test_feature_importances_are_populated_and_sorted(forecast_result):
    importances = forecast_result.feature_importances

    assert importances
    values = list(importances.values())
    assert values == sorted(values, reverse=True)
    assert all(v >= 0 for v in values)


def test_forecast_raises_when_not_enough_history():
    rows = []
    for day in range(5):
        rows.append(
            {
                Col.TRANSACTION_ID: f"T{day}",
                Col.DATE: pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                Col.PRODUCT_ID: "SKU-ONLY",
                Col.QUANTITY: 1.0,
                Col.UNIT_PRICE: 5.0,
                Col.LINE_TOTAL: 5.0,
            }
        )
    df = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="Not enough history"):
        forecast(df, ForecastParams())


def test_product_store_granularity(forecast_transactions: pd.DataFrame):
    df = forecast_transactions.copy()
    df[Col.STORE_ID] = "STORE-1"
    params = ForecastParams(horizon_days=7, granularity="product_store", quantiles=(0.5,))

    result = forecast(df, params)

    assert result.granularity == "product_store"
    assert result.n_series == 4
    for series in result.series:
        assert series.store_id == "STORE-1"
        assert series.series_id == f"{series.product_id}|STORE-1"
