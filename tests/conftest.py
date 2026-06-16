"""Synthetic transaction fixtures with planted signal, shared across `ml/`
and `api/` tests.

Each fixture returns a DataFrame already in the canonical post-validation
shape (see `ml.contracts.Col`): typed columns, `line_total` derived, dates as
`datetime64`. This lets `ml/` unit tests call module entrypoints directly
without going through `ml.validation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.contracts import Col


def _to_clean_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[Col.DATE] = pd.to_datetime(df[Col.DATE])
    df[Col.QUANTITY] = df[Col.QUANTITY].astype(float)
    df[Col.UNIT_PRICE] = df[Col.UNIT_PRICE].astype(float)
    df[Col.LINE_TOTAL] = df[Col.QUANTITY] * df[Col.UNIT_PRICE]
    return df


@pytest.fixture(scope="session")
def forecast_transactions() -> pd.DataFrame:
    """~150 days x 3 products with a clear weekly seasonality + linear trend
    + periodic promos (price cut + demand spike), plus one product introduced
    only in the last 10 days (cold-start / `low_confidence` case).
    """
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2024-01-01")
    n_days = 150
    dates = pd.date_range(start, periods=n_days, freq="D")

    products = {
        "SKU-A": {"base": 20.0, "price": 10.0, "category": "Beverages"},
        "SKU-B": {"base": 12.0, "price": 5.0, "category": "Snacks"},
        "SKU-C": {"base": 8.0, "price": 25.0, "category": "Household"},
    }

    rows: list[dict] = []
    txn = 0
    for day_idx, d in enumerate(dates):
        is_weekend = d.dayofweek >= 5
        is_promo = (day_idx % 14) < 3
        for product_id, cfg in products.items():
            demand = cfg["base"] + 0.05 * day_idx
            if is_weekend:
                demand *= 1.5
            price = cfg["price"]
            if is_promo:
                demand *= 2.0
                price *= 0.8
            demand += rng.normal(0, 1.0)
            qty = max(round(demand), 0)
            if qty <= 0:
                continue
            txn += 1
            rows.append(
                {
                    Col.TRANSACTION_ID: f"F{txn}",
                    Col.DATE: d,
                    Col.PRODUCT_ID: product_id,
                    Col.QUANTITY: float(qty),
                    Col.UNIT_PRICE: price,
                    Col.CATEGORY: cfg["category"],
                }
            )

    # Cold-start product: only sold in the last 10 days.
    for d in dates[-10:]:
        txn += 1
        rows.append(
            {
                Col.TRANSACTION_ID: f"F{txn}",
                Col.DATE: d,
                Col.PRODUCT_ID: "SKU-NEW",
                Col.QUANTITY: 5.0,
                Col.UNIT_PRICE: 3.0,
                Col.CATEGORY: "New",
            }
        )

    return _to_clean_df(rows)


@pytest.fixture(scope="session")
def basket_transactions() -> pd.DataFrame:
    """~400 baskets. ITEM-A + ITEM-B co-occur in ~40% of baskets and ITEM-C +
    ITEM-D in ~30% - planted high-lift pairs against a noise pool.
    """
    rng = np.random.default_rng(7)
    noise_pool = ["ITEM-E", "ITEM-F", "ITEM-G", "ITEM-H", "ITEM-I"]
    base_date = pd.Timestamp("2024-01-01")

    rows: list[dict] = []
    for i in range(400):
        basket: set[str] = set()
        r = rng.random()
        if r < 0.4:
            basket.update({"ITEM-A", "ITEM-B"})
        elif r < 0.7:
            basket.update({"ITEM-C", "ITEM-D"})

        n_extra = int(rng.integers(0, 3))
        if n_extra:
            basket.update(rng.choice(noise_pool, size=n_extra, replace=False))
        if not basket:
            basket.add(str(rng.choice(noise_pool)))

        date = base_date + pd.Timedelta(days=int(rng.integers(0, 60)))
        for item in basket:
            rows.append(
                {
                    Col.TRANSACTION_ID: f"B{i}",
                    Col.DATE: date,
                    Col.PRODUCT_ID: item,
                    Col.QUANTITY: 1.0,
                    Col.UNIT_PRICE: 5.0,
                    Col.CATEGORY: "General",
                }
            )

    return _to_clean_df(rows)


@pytest.fixture(scope="session")
def customer_transactions() -> pd.DataFrame:
    """Two cohorts over a 200-day observation window: 10 "loyal" customers
    buying every ~10 days throughout, and 10 "lapsed" customers who only
    bought a handful of times in the first 30 days.
    """
    rng = np.random.default_rng(99)
    base_date = pd.Timestamp("2024-01-01")
    observation_days = 200

    rows: list[dict] = []
    txn = 0
    for c in range(10):
        customer_id = f"LOYAL-{c}"
        for day in range(0, observation_days, 10):
            txn += 1
            rows.append(
                {
                    Col.TRANSACTION_ID: f"C{txn}",
                    Col.DATE: base_date + pd.Timedelta(days=day),
                    Col.PRODUCT_ID: "SKU-X",
                    Col.CUSTOMER_ID: customer_id,
                    Col.QUANTITY: 2.0,
                    Col.UNIT_PRICE: 25.0 + float(rng.normal(0, 2)),
                }
            )

    for c in range(10):
        customer_id = f"LAPSED-{c}"
        for day in range(0, 30, 10):
            txn += 1
            rows.append(
                {
                    Col.TRANSACTION_ID: f"C{txn}",
                    Col.DATE: base_date + pd.Timedelta(days=day),
                    Col.PRODUCT_ID: "SKU-X",
                    Col.CUSTOMER_ID: customer_id,
                    Col.QUANTITY: 1.0,
                    Col.UNIT_PRICE: 25.0 + float(rng.normal(0, 2)),
                }
            )

    return _to_clean_df(rows)


@pytest.fixture(scope="session")
def api_transactions_csv() -> bytes:
    """A small but valid raw CSV (with header aliases) for the dataset-upload
    pipeline: >=29 days of history (forecast), customer ids (customer
    module), and a planted basket pair.
    """
    rng = np.random.default_rng(123)
    base_date = pd.Timestamp("2024-01-01")
    customers = [f"CUST-{i}" for i in range(8)]

    lines = ["order_id,order_date,sku,customerid,qty,price,category"]
    txn = 0
    for day in range(40):
        date = (base_date + pd.Timedelta(days=day)).date().isoformat()
        for _ in range(6):
            txn += 1
            customer = str(rng.choice(customers))
            # ITEM-A and ITEM-B co-occur often (separate "transactions" sharing
            # an order id so the basket module sees them as one basket).
            if rng.random() < 0.5:
                for item, price in (("ITEM-A", "4.5"), ("ITEM-B", "3.0")):
                    lines.append(f"O{txn},{date},{item},{customer},1,{price},General")
            else:
                item = str(rng.choice(["ITEM-C", "ITEM-D", "ITEM-E"]))
                lines.append(f"O{txn},{date},{item},{customer},1,5.0,General")

    return ("\n".join(lines) + "\n").encode("utf-8")
