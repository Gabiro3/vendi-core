"""Iso-value segmentation and contour grid (Fader-Hardie iso-value framework).

Two customers with different RFM histories can have the same predicted future
value - they sit on the same "iso-value curve." `segment_customers` ranks
customers by predicted CLV and splits them into `n_segments` bands; each band
maps to a retention action. `build_iso_value_grid` sweeps a frequency x
recency grid (holding T and monetary_value at representative values) and
returns predicted CLV at each point - the literal iso-value contours, for a
future dashboard to render.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.contracts import CustomerParams, IsoValueGridPoint

# Customer is flagged `is_at_risk_high_value` when in the top value segment
# AND their probability of still being active has dropped below this.
AT_RISK_PROB_ALIVE_THRESHOLD = 0.5

_NAMED_SEGMENT_LABELS = ["Minimal", "Low", "Mid", "High", "Top value"]

_SEGMENT_ACTIONS: dict[str, str] = {
    "Minimal": "Low priority - minimal investment.",
    "Low": "Nurture with low-cost, automated engagement.",
    "Mid": "Nurture - targeted offers to grow toward higher value.",
    "High": "Reward and retain - prioritize for loyalty programs.",
    "Top value": "Retain and reward - VIP treatment, proactive outreach.",
}

_GRID_SIZE = 20


def segment_labels(n_segments: int) -> list[str]:
    """Band labels, lowest-CLV to highest-CLV. For the default of 5, uses the
    named bands from the spec; otherwise generic "Tier N" labels.
    """
    if n_segments == len(_NAMED_SEGMENT_LABELS):
        return list(_NAMED_SEGMENT_LABELS)
    return [f"Tier {i + 1}" for i in range(n_segments)]


def segment_action(label: str) -> str:
    return _SEGMENT_ACTIONS.get(label, "Review and assign a retention strategy.")


def segment_customers(clv: pd.Series, n_segments: int) -> tuple[pd.Series, list[str], list[str]]:
    """Split customers into `n_segments` quantile bands by predicted CLV.

    Returns (segment_labels_per_customer, ordered_labels, warnings). Ranking
    before `qcut` guarantees monotonic, equal-sized bands even with many tied
    CLV values (e.g. many customers at CLV=0).
    """
    warnings: list[str] = []
    n_customers = len(clv)
    if n_customers == 0:
        return pd.Series(dtype="object"), [], warnings

    effective_n = min(n_segments, n_customers)
    if effective_n < n_segments:
        warnings.append(
            f"Only {n_customers} customers; reduced value segments from "
            f"{n_segments} to {effective_n}."
        )

    labels = segment_labels(effective_n)
    ranks = clv.rank(method="first")
    bands = pd.qcut(ranks, effective_n, labels=labels)
    return bands.astype(str), labels, warnings


def build_iso_value_grid(
    bgf,
    ggf,
    summary: pd.DataFrame,
    params: CustomerParams,
    grid_size: int = _GRID_SIZE,
) -> list[IsoValueGridPoint]:
    """Sweep frequency x recency, holding T and monetary_value at their
    (repeat-customer) medians, and return predicted CLV at each grid point.
    """
    T_rep = float(summary["T"].median()) if len(summary) else 0.0
    repeat = summary[summary["monetary_value"] > 0]
    monetary_rep = float(repeat["monetary_value"].median()) if len(repeat) else 0.0

    max_freq = max(float(summary["frequency"].max()) if len(summary) else 0.0, 1.0)
    freq_grid = np.linspace(0, max_freq, grid_size)
    recency_grid = np.linspace(0, T_rep, grid_size)
    freq_mesh, recency_mesh = np.meshgrid(freq_grid, recency_grid)
    freq_flat = freq_mesh.ravel()
    recency_flat = recency_mesh.ravel()
    T_flat = np.full_like(freq_flat, T_rep)
    monetary_flat = np.full_like(freq_flat, monetary_rep)

    if ggf is not None and monetary_rep > 0:
        clv_flat = ggf.customer_lifetime_value(
            bgf,
            pd.Series(freq_flat),
            pd.Series(recency_flat),
            pd.Series(T_flat),
            pd.Series(monetary_flat),
            time=params.forecast_period_days / 30.0,
            discount_rate=params.discount_rate,
            freq="D",
        ).to_numpy()
    else:
        predicted = bgf.conditional_expected_number_of_purchases_up_to_time(
            params.forecast_period_days, freq_flat, recency_flat, T_flat
        )
        clv_flat = np.asarray(predicted) * monetary_rep

    # Gamma-Gamma's unconditional mean (used at frequency=0) is undefined/negative
    # when the fitted `q` shape parameter is <= 1, producing NaN or large negative
    # values; CLV can't be negative in practice, so floor at 0 rather than surface
    # those artifacts.
    clv_flat = np.nan_to_num(clv_flat, nan=0.0)
    clv_flat = np.clip(clv_flat, 0.0, None)

    return [
        IsoValueGridPoint(frequency=float(f), recency=float(r), clv=float(c))
        for f, r, c in zip(freq_flat, recency_flat, clv_flat, strict=True)
    ]
