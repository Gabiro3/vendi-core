"""Market-basket analysis: FP-Growth + association rules, optionally sliced.

`analyze_baskets(df, params) -> BasketResult` is the sole entrypoint. Pure
pandas/mlxtend; no DB or HTTP knowledge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from ml.contracts import BasketParams, BasketResult, BasketRule, Col, SliceCoverage

# Conviction is `inf` when confidence == 1.0. Cap it so results stay
# JSON-safe and comparable.
_MAX_CONVICTION = 1_000.0


def analyze_baskets(df: pd.DataFrame, params: BasketParams) -> BasketResult:
    warnings: list[str] = []

    if params.level == "category" and Col.CATEGORY not in df.columns:
        raise ValueError("level='category' requires a 'category' column in the dataset")
    item_col = Col.CATEGORY if params.level == "category" else Col.PRODUCT_ID

    work = df.copy()
    work[Col.DATE] = pd.to_datetime(work[Col.DATE])

    dimension = params.dimension
    if dimension == "store" and Col.STORE_ID not in work.columns:
        warnings.append(
            "dimension='store' requested but no store_id column is present; ignoring dimension."
        )
        dimension = None
    if dimension == "time_of_day" and (work[Col.DATE].dt.hour == 0).all():
        warnings.append(
            "dimension='time_of_day' requested but the dataset has no time-of-day "
            "information; ignoring dimension."
        )
        dimension = None

    work["_slice"] = _compute_slice_key(work, dimension)

    if item_col == Col.CATEGORY:
        work = work.dropna(subset=[Col.CATEGORY])

    itemsets_by_txn = (
        work.groupby([Col.TRANSACTION_ID, "_slice"], dropna=False)[item_col]
        .apply(lambda s: frozenset(str(v) for v in s.dropna().unique()))
        .reset_index(name="itemset")
    )
    itemsets_by_txn = itemsets_by_txn[itemsets_by_txn["itemset"].map(len) > 0]

    coverage: list[SliceCoverage] = []
    all_rules: list[BasketRule] = []

    for slice_key, slice_group in itemsets_by_txn.groupby("_slice", dropna=False):
        itemsets = slice_group["itemset"].tolist()
        rules_df, n_itemsets = _run_fpgrowth(
            itemsets, params.min_support, params.min_confidence, params.min_lift, params.max_len
        )
        rules_df = _dedupe_symmetric_pairs(rules_df)

        key = slice_key if pd.notna(slice_key) else None
        coverage.append(
            SliceCoverage(
                slice_key=key,
                n_transactions=n_itemsets,
                n_itemsets=n_itemsets,
                n_rules_before_cap=len(rules_df),
            )
        )

        for _, row in rules_df.iterrows():
            conviction = float(row["conviction"])
            if not np.isfinite(conviction):
                conviction = _MAX_CONVICTION
            all_rules.append(
                BasketRule(
                    antecedents=sorted(row["antecedents"]),
                    consequents=sorted(row["consequents"]),
                    support=float(row["support"]),
                    confidence=float(row["confidence"]),
                    lift=float(row["lift"]),
                    leverage=float(row["leverage"]),
                    conviction=conviction,
                    decision=_decision_text(float(row["confidence"]), float(row["lift"])),
                    slice_key=key,
                )
            )

    all_rules.sort(key=lambda r: r.lift, reverse=True)

    if not all_rules:
        warnings.append(
            "No association rules met the given thresholds; try lowering "
            "min_support, min_confidence, or min_lift."
        )

    total_rules_found = len(all_rules)
    rules_truncated = total_rules_found > params.max_rules
    if rules_truncated:
        warnings.append(
            f"{total_rules_found} rules met the thresholds; returning the top "
            f"{params.max_rules} by lift."
        )

    return BasketResult(
        rules=all_rules[: params.max_rules],
        total_rules_found=total_rules_found,
        rules_truncated=rules_truncated,
        coverage=coverage,
        level=params.level,
        dimension=dimension,
        warnings=warnings,
        overflow_rules=all_rules[params.max_rules :],
    )


def _compute_slice_key(work: pd.DataFrame, dimension: str | None) -> pd.Series:
    if dimension is None:
        return pd.Series([None] * len(work), index=work.index, dtype=object)
    if dimension == "store":
        return "store=" + work[Col.STORE_ID].astype(str)
    if dimension == "day_of_week":
        return "day_of_week=" + work[Col.DATE].dt.day_name()
    if dimension == "time_of_day":
        hour = work[Col.DATE].dt.hour
        bucket = pd.cut(
            hour, bins=[-1, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"]
        )
        return "time_of_day=" + bucket.astype(str)
    raise ValueError(f"Unknown dimension: {dimension!r}")


def _run_fpgrowth(
    itemsets: list[frozenset[str]],
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_len: int,
) -> tuple[pd.DataFrame, int]:
    n_itemsets = len(itemsets)
    if n_itemsets == 0:
        return pd.DataFrame(), 0

    encoder = TransactionEncoder()
    onehot = encoder.fit(itemsets).transform(itemsets)
    onehot_df = pd.DataFrame(onehot, columns=encoder.columns_)

    frequent = fpgrowth(onehot_df, min_support=min_support, use_colnames=True, max_len=max_len)
    if frequent.empty:
        return pd.DataFrame(), n_itemsets

    rules = association_rules(
        frequent, num_itemsets=len(onehot_df), metric="lift", min_threshold=min_lift
    )
    if rules.empty:
        return rules, n_itemsets

    rules = rules[
        (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
        & (rules["support"] >= min_support)
    ]
    return rules, n_itemsets


def _dedupe_symmetric_pairs(rules: pd.DataFrame) -> pd.DataFrame:
    """For simple 1-item -> 1-item rules, A->B and B->A are mirror images with
    identical support/lift but different confidence. Keep only the
    higher-confidence direction (the more actionable recommendation).
    """
    if rules.empty:
        return rules

    keep = pd.Series(True, index=rules.index)
    best_for_pair: dict[frozenset, int] = {}
    for idx, row in rules.iterrows():
        antecedents, consequents = row["antecedents"], row["consequents"]
        if len(antecedents) != 1 or len(consequents) != 1:
            continue
        pair_key = frozenset(antecedents | consequents)
        existing = best_for_pair.get(pair_key)
        if existing is None:
            best_for_pair[pair_key] = idx
            continue
        if row["confidence"] > rules.loc[existing, "confidence"]:
            keep[existing] = False
            best_for_pair[pair_key] = idx
        else:
            keep[idx] = False

    return rules[keep]


def _decision_text(confidence: float, lift: float) -> str:
    if lift >= 3.0 and confidence >= 0.5:
        return "Strong pairing - bundle as a combo offer or cross-promote at checkout."
    if lift >= 2.0:
        return "Notable affinity - place these items near each other on the shelf."
    if confidence >= 0.4:
        return "Moderate affinity - consider a 'frequently bought together' promotion."
    return "Weak affinity - worth monitoring as more data accumulates."
