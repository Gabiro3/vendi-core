from __future__ import annotations

from datetime import date

import pandas as pd

from ml.contracts import Col
from ml.validation import ValidationConfig, normalize_header, validate_transactions


def test_normalize_header_collapses_punctuation_and_case():
    assert normalize_header("Order ID") == "order_id"
    assert normalize_header(" Unit-Price ") == "unit_price"
    assert normalize_header("Qty") == "qty"


def test_validate_transactions_happy_path_maps_aliases_and_derives_line_total():
    raw = pd.DataFrame(
        {
            "Order ID": ["T1", "T2"],
            "Order Date": ["2024-01-01", "2024-01-02"],
            "SKU": ["A", "B"],
            "Qty": ["2", "3"],
            "Unit Price": ["10.0", "5.5"],
        }
    )
    config = ValidationConfig(min_transactions=1, min_date_span_weeks=1, today=date(2024, 6, 1))
    clean, report = validate_transactions(raw, config)

    assert report.is_valid
    assert report.errors == []
    assert report.rows_in == 2
    assert report.rows_kept == 2
    assert report.rows_dropped == 0
    assert report.has_customer_id is False
    assert list(clean[Col.LINE_TOTAL]) == [20.0, 16.5]
    assert report.date_start == date(2024, 1, 1)
    assert report.date_end == date(2024, 1, 2)


def test_validate_transactions_reports_missing_required_columns():
    raw = pd.DataFrame({"order_id": ["T1"], "sku": ["A"]})
    clean, report = validate_transactions(raw)

    assert not report.is_valid
    assert clean.empty
    assert report.errors
    assert "quantity" in report.errors[0]


def test_validate_transactions_drops_bad_rows_and_dedupes():
    raw = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T5"],
            "date": ["2024-01-01", "2024-01-02", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-04"],
            "product_id": ["A", "B", "C", "D", "E", "E"],
            "quantity": ["2", "-1", "1", "0", "1", "1"],
            "unit_price": ["10.0", "5.0", "5.0", "5.0", "3.0", "3.0"],
        }
    )
    config = ValidationConfig(min_transactions=1, min_date_span_weeks=1, today=date(2024, 6, 1))
    clean, report = validate_transactions(raw, config)

    assert report.is_valid
    # T2 (negative qty) and T4 (zero qty) dropped as non-positive quantity,
    # T3 dropped for an invalid date, T5 deduped (duplicate of itself).
    assert report.rows_dropped == 4
    assert report.duplicate_rows_removed == 1
    assert report.dropped_by_reason["non_positive_quantity"] == 2
    assert report.dropped_by_reason["invalid_date"] == 1
    assert report.rows_kept == 2
    assert set(clean[Col.PRODUCT_ID]) == {"A", "E"}


def test_validate_transactions_warns_on_small_dataset():
    raw = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2"],
            "date": ["2024-01-01", "2024-01-02"],
            "product_id": ["A", "B"],
            "quantity": ["1", "1"],
            "unit_price": ["1.0", "1.0"],
        }
    )
    config = ValidationConfig(min_transactions=200, min_date_span_weeks=8, today=date(2024, 6, 1))
    _, report = validate_transactions(raw, config)

    assert report.is_valid
    assert any("distinct transactions" in w for w in report.warnings)
    assert any("Date range" in w for w in report.warnings)
    assert any("customer_id" in w for w in report.warnings)
