"""Backtesting and metrics for the forecasting module.

Time-based holdout: train on data before the holdout window, evaluate the
trained tweedie model on the holdout using its precomputed (actual-history)
features. This is a one-step-ahead style evaluation rather than a fully
recursive backtest - cheaper, and standard practice for model selection.
Always compared against a seasonal-naive (lag-7) baseline via MASE and RMSE.

WRMSSE (the M5 competition's hierarchical metric) is the more rigorous option
referenced by the spec but not implemented in v1.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.contracts import BacktestMetrics, Col, ForecastParams
from ml.features import DEFAULT_LAGS


def train_lgbm_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    objective: str,
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM regressor.

    `objective='tweedie'` for the point forecast (handles zero-heavy
    intermittent demand); `objective='quantile', alpha=q` for prediction
    intervals.
    """
    lgbm_params: dict = {
        "objective": objective,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
    }
    if objective == "quantile":
        if alpha is None:
            raise ValueError("alpha is required for objective='quantile'")
        lgbm_params["alpha"] = alpha
    if objective == "tweedie":
        lgbm_params["tweedie_variance_power"] = 1.3

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X, y, categorical_feature=categorical_cols or "auto")
    return model


def run_backtest(
    featured: pd.DataFrame,
    categorical_cols: list[str],
    model_feature_cols: list[str],
    params: ForecastParams,
) -> tuple[BacktestMetrics, list[str]]:
    """Hold out the last `horizon_days * backtest_windows` days (capped to
    the available history) and report MASE/RMSE for the model vs. a
    seasonal-naive (lag-7) baseline.
    """
    warnings: list[str] = []
    max_lag = max(DEFAULT_LAGS)
    clean = featured.dropna(subset=[f"{Col.QUANTITY}_lag_{max_lag}", f"{Col.QUANTITY}_lag_7"])

    if clean.empty:
        return _empty_metrics(), ["Not enough history to backtest."]

    dates = np.sort(clean[Col.DATE].unique())
    test_size = params.horizon_days * params.backtest_windows
    if len(dates) <= test_size + 7:
        test_size = max(min(params.horizon_days, len(dates) - 1), 1)
        warnings.append("Limited history: reduced the backtest window size.")

    if test_size <= 0 or test_size >= len(dates):
        return _empty_metrics(), ["Not enough history to backtest."]

    split_date = dates[-test_size]
    train = clean[clean[Col.DATE] < split_date]
    test = clean[clean[Col.DATE] >= split_date]

    if train.empty or test.empty:
        return _empty_metrics(), ["Not enough history to backtest."]

    X_train, y_train = train[model_feature_cols], train[Col.QUANTITY]
    X_test, y_test = test[model_feature_cols], test[Col.QUANTITY]

    model = train_lgbm_model(X_train, y_train, categorical_cols, objective="tweedie")
    preds = np.clip(model.predict(X_test), 0, None)
    baseline_preds = test[f"{Col.QUANTITY}_lag_7"].fillna(0.0).to_numpy()
    y_test_arr = y_test.to_numpy()

    rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((y_test_arr - baseline_preds) ** 2)))

    # MASE scale = in-sample MAE of the seasonal-naive (lag-7) baseline.
    train_actual = train[Col.QUANTITY].to_numpy()
    train_seasonal_naive = train[f"{Col.QUANTITY}_lag_7"].to_numpy()
    valid = ~np.isnan(train_seasonal_naive)
    scale = (
        float(np.mean(np.abs(train_actual[valid] - train_seasonal_naive[valid])))
        if valid.any()
        else 0.0
    )

    mase: float | None
    baseline_mase: float | None
    if scale > 0:
        mase = float(np.mean(np.abs(y_test_arr - preds)) / scale)
        baseline_mase = float(np.mean(np.abs(y_test_arr - baseline_preds)) / scale)
    else:
        mase = None
        baseline_mase = None
        warnings.append("Could not compute MASE (no variation in the seasonal-naive baseline).")

    beats_baseline = (
        mase < baseline_mase if mase is not None and baseline_mase is not None else None
    )
    if beats_baseline is False:
        warnings.append("Model did not beat the seasonal-naive baseline on the backtest window.")

    n_windows = max(1, test_size // params.horizon_days)

    return (
        BacktestMetrics(
            mase=mase,
            rmse=rmse,
            baseline_mase=baseline_mase,
            baseline_rmse=baseline_rmse,
            beats_baseline=beats_baseline,
            n_windows=n_windows,
        ),
        warnings,
    )


def _empty_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        mase=None,
        rmse=None,
        baseline_mase=None,
        baseline_rmse=None,
        beats_baseline=None,
        n_windows=0,
    )
