from __future__ import annotations

import pandas as pd
import pytest

from ml.contracts import Col, CustomerParams
from ml.customer.clv import model_customers


def test_model_customers_ranks_loyal_above_lapsed(customer_transactions: pd.DataFrame):
    params = CustomerParams(forecast_period_days=365, n_value_segments=5, include_iso_value_grid=True)
    result = model_customers(customer_transactions, params)

    by_id = {r.customer_id: r for r in result.records}
    loyal_clv = [by_id[f"LOYAL-{c}"].clv for c in range(10)]
    lapsed_clv = [by_id[f"LAPSED-{c}"].clv for c in range(10)]
    loyal_prob_alive = [by_id[f"LOYAL-{c}"].prob_alive for c in range(10)]
    lapsed_prob_alive = [by_id[f"LAPSED-{c}"].prob_alive for c in range(10)]

    assert min(loyal_clv) > max(lapsed_clv)
    assert min(loyal_prob_alive) > max(lapsed_prob_alive)

    # Loyal customers bought every 10 days for ~190 days -> frequency ~19.
    for c in range(10):
        assert by_id[f"LOYAL-{c}"].frequency > by_id[f"LAPSED-{c}"].frequency

    assert result.n_customers == 20
    assert result.n_repeat_customers == 20
    assert result.gamma_gamma_params is not None
    assert result.gamma_gamma_assumption_ok in (True, False)


def test_segment_summary_monotonic_in_clv(customer_transactions: pd.DataFrame):
    params = CustomerParams(n_value_segments=5)
    result = model_customers(customer_transactions, params)

    assert len(result.segment_summary) == 5
    avg_clvs = [s.avg_clv for s in result.segment_summary]
    assert avg_clvs == sorted(avg_clvs)
    assert sum(s.count for s in result.segment_summary) == result.n_customers

    labels = [s.segment for s in result.segment_summary]
    assert labels == ["Minimal", "Low", "Mid", "High", "Top value"]


def test_iso_value_grid_is_populated(customer_transactions: pd.DataFrame):
    params = CustomerParams(include_iso_value_grid=True)
    result = model_customers(customer_transactions, params)

    assert result.iso_value_grid is not None
    assert len(result.iso_value_grid) == 400
    assert all(p.clv >= 0 for p in result.iso_value_grid)


def test_iso_value_grid_omitted_when_disabled(customer_transactions: pd.DataFrame):
    params = CustomerParams(include_iso_value_grid=False)
    result = model_customers(customer_transactions, params)

    assert result.iso_value_grid is None


def test_model_customers_requires_customer_id_column(forecast_transactions: pd.DataFrame):
    assert Col.CUSTOMER_ID not in forecast_transactions.columns
    with pytest.raises(ValueError, match="customer_id"):
        model_customers(forecast_transactions, CustomerParams())


def test_gamma_gamma_fallback_with_fewer_than_two_eligible_customers():
    rows = [
        {
            Col.TRANSACTION_ID: "T1",
            Col.DATE: pd.Timestamp("2024-01-01"),
            Col.PRODUCT_ID: "SKU-X",
            Col.CUSTOMER_ID: "A",
            Col.QUANTITY: 1.0,
            Col.UNIT_PRICE: 10.0,
            Col.LINE_TOTAL: 10.0,
        },
        {
            Col.TRANSACTION_ID: "T2",
            Col.DATE: pd.Timestamp("2024-01-10"),
            Col.PRODUCT_ID: "SKU-X",
            Col.CUSTOMER_ID: "A",
            Col.QUANTITY: 1.0,
            Col.UNIT_PRICE: 10.0,
            Col.LINE_TOTAL: 10.0,
        },
        {
            Col.TRANSACTION_ID: "T3",
            Col.DATE: pd.Timestamp("2024-01-05"),
            Col.PRODUCT_ID: "SKU-X",
            Col.CUSTOMER_ID: "B",
            Col.QUANTITY: 1.0,
            Col.UNIT_PRICE: 10.0,
            Col.LINE_TOTAL: 10.0,
        },
    ]
    df = pd.DataFrame(rows)
    result = model_customers(df, CustomerParams(n_value_segments=2))

    assert result.gamma_gamma_params is None
    assert result.gamma_gamma_corr == 0.0
    assert any("Gamma-Gamma" in w for w in result.warnings)
    assert result.n_customers == 2
    assert result.n_repeat_customers == 1
