"""Customer modeling: BG/NBD (transactions) + Gamma-Gamma (monetary) -> CLV,
plus iso-value segmentation.

`model_customers(df, params) -> CustomerResult` is the sole entrypoint.
"""

from __future__ import annotations

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from ml.contracts import Col, CustomerParams, CustomerRecord, CustomerResult, SegmentSummary
from ml.customer.isovalue import (
    AT_RISK_PROB_ALIVE_THRESHOLD,
    build_iso_value_grid,
    segment_action,
    segment_customers,
)


def model_customers(df: pd.DataFrame, params: CustomerParams) -> CustomerResult:
    warnings: list[str] = []

    if Col.CUSTOMER_ID not in df.columns:
        raise ValueError("Customer module requires a 'customer_id' column")

    work = df.copy()
    work[Col.DATE] = pd.to_datetime(work[Col.DATE])
    work = work.dropna(subset=[Col.CUSTOMER_ID])
    if work.empty:
        raise ValueError("No rows with a customer_id; cannot run the customer module")

    observation_end = work[Col.DATE].max()

    summary = summary_data_from_transaction_data(
        work,
        customer_id_col=Col.CUSTOMER_ID,
        datetime_col=Col.DATE,
        monetary_value_col=Col.LINE_TOTAL,
        observation_period_end=observation_end,
        freq="D",
    )

    n_customers = len(summary)
    observation_period_days = int(round(summary["T"].max())) if n_customers else 0

    bgf = BetaGeoFitter(penalizer_coef=0.5)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    # On tiny/degenerate datasets the BG/NBD fit can push a/b towards 0, which
    # makes this expectation NaN for frequency=0 customers; floor those at 0
    # expected future purchases rather than letting NaN propagate into CLV.
    summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        params.forecast_period_days, summary["frequency"], summary["recency"], summary["T"]
    ).fillna(0.0)
    summary["prob_alive"] = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )

    eligible = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    n_repeat_customers = int((summary["frequency"] > 0).sum())

    gamma_gamma_corr = 0.0
    gamma_gamma_assumption_ok = True
    gamma_gamma_params: dict[str, float] | None = None
    ggf: GammaGammaFitter | None = None

    if len(eligible) >= 2:
        gamma_gamma_corr = float(eligible["frequency"].corr(eligible["monetary_value"]))
        gamma_gamma_assumption_ok = abs(gamma_gamma_corr) < 0.3
        if not gamma_gamma_assumption_ok:
            warnings.append(
                "Gamma-Gamma independence assumption may not hold: Pearson r="
                f"{gamma_gamma_corr:.2f} between frequency and monetary value "
                "(expected close to 0). CLV monetary estimates may be biased."
            )

        ggf = GammaGammaFitter(penalizer_coef=1.0)
        ggf.fit(eligible["frequency"], eligible["monetary_value"])
        gamma_gamma_params = {k: float(v) for k, v in ggf.params_.items()}

        summary["predicted_avg_value"] = ggf.conditional_expected_average_profit(
            summary["frequency"], summary["monetary_value"]
        )
        summary["clv"] = ggf.customer_lifetime_value(
            bgf,
            summary["frequency"],
            summary["recency"],
            summary["T"],
            summary["monetary_value"],
            time=params.forecast_period_days / 30.0,
            discount_rate=params.discount_rate,
            freq="D",
        )
    else:
        warnings.append(
            "Fewer than 2 customers with repeat purchases and positive spend; "
            "Gamma-Gamma monetary model skipped. CLV is approximated as "
            "predicted purchases x average basket value."
        )
        avg_basket_value = _average_basket_value(work)
        summary["predicted_avg_value"] = avg_basket_value
        summary["clv"] = summary["predicted_purchases"] * avg_basket_value

    value_segments, ordered_labels, segment_warnings = segment_customers(
        summary["clv"], params.n_value_segments
    )
    warnings.extend(segment_warnings)
    summary["value_segment"] = value_segments.values

    top_segment = ordered_labels[-1] if ordered_labels else None
    summary["is_at_risk_high_value"] = (summary["value_segment"] == top_segment) & (
        summary["prob_alive"] < AT_RISK_PROB_ALIVE_THRESHOLD
    )

    records = [
        CustomerRecord(
            customer_id=str(customer_id),
            frequency=float(row["frequency"]),
            recency=float(row["recency"]),
            T=float(row["T"]),
            monetary_value=float(row["monetary_value"]),
            predicted_purchases=float(row["predicted_purchases"]),
            prob_alive=float(row["prob_alive"]),
            predicted_avg_value=float(row["predicted_avg_value"]),
            clv=float(row["clv"]),
            value_segment=str(row["value_segment"]),
            action=segment_action(str(row["value_segment"])),
            is_at_risk_high_value=bool(row["is_at_risk_high_value"]),
        )
        for customer_id, row in summary.iterrows()
    ]

    total_clv = float(summary["clv"].sum())
    segment_summary = []
    for label in ordered_labels:
        band = summary[summary["value_segment"] == label]
        band_clv = float(band["clv"].sum())
        segment_summary.append(
            SegmentSummary(
                segment=label,
                count=int(len(band)),
                avg_clv=float(band["clv"].mean()) if len(band) else 0.0,
                pct_of_total_value=(band_clv / total_clv * 100.0) if total_clv else 0.0,
            )
        )

    iso_value_grid = None
    if params.include_iso_value_grid and n_customers:
        iso_value_grid = build_iso_value_grid(bgf, ggf, summary, params)

    bgnbd_params = {k: float(v) for k, v in bgf.params_.items()}

    return CustomerResult(
        records=records,
        segment_summary=segment_summary,
        gamma_gamma_corr=gamma_gamma_corr,
        gamma_gamma_assumption_ok=gamma_gamma_assumption_ok,
        bgnbd_params=bgnbd_params,
        gamma_gamma_params=gamma_gamma_params,
        iso_value_grid=iso_value_grid,
        observation_period_days=observation_period_days,
        forecast_period_days=params.forecast_period_days,
        n_customers=n_customers,
        n_repeat_customers=n_repeat_customers,
        warnings=warnings,
    )


def _average_basket_value(df: pd.DataFrame) -> float:
    per_transaction = df.groupby(Col.TRANSACTION_ID)[Col.LINE_TOTAL].sum()
    return float(per_transaction.mean()) if len(per_transaction) else 0.0
