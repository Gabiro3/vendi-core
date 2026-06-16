from __future__ import annotations

import pandas as pd

from ml.basket.rules import analyze_baskets
from ml.contracts import BasketParams, BasketRule


def _pair_rules(rules: list[BasketRule], a: str, b: str) -> list[BasketRule]:
    return [
        r
        for r in rules
        if set(r.antecedents) | set(r.consequents) == {a, b}
    ]


def test_analyze_baskets_finds_planted_high_lift_pairs(basket_transactions: pd.DataFrame):
    params = BasketParams(min_support=0.01, min_confidence=0.1, min_lift=1.0, max_rules=50)
    result = analyze_baskets(basket_transactions, params)

    assert result.total_rules_found > 0
    assert not result.rules_truncated

    ab_rules = _pair_rules(result.rules, "ITEM-A", "ITEM-B")
    cd_rules = _pair_rules(result.rules, "ITEM-C", "ITEM-D")
    assert ab_rules, "expected an ITEM-A/ITEM-B rule"
    assert cd_rules, "expected an ITEM-C/ITEM-D rule"
    assert all(r.lift > 1.2 for r in ab_rules)
    assert all(r.lift > 1.2 for r in cd_rules)

    # Planted pairs should outrank noise: top rule by lift is one of the two.
    top = result.rules[0]
    assert set(top.antecedents) | set(top.consequents) in ({"ITEM-A", "ITEM-B"}, {"ITEM-C", "ITEM-D"})

    # Each rule has a non-empty decision string and a single coverage slice.
    assert all(r.decision for r in result.rules)
    assert len(result.coverage) == 1
    assert result.coverage[0].slice_key is None
    assert result.coverage[0].n_transactions == 400


def test_analyze_baskets_dedupes_symmetric_pairs(basket_transactions: pd.DataFrame):
    params = BasketParams(min_support=0.01, min_confidence=0.1, min_lift=1.0, max_rules=50)
    result = analyze_baskets(basket_transactions, params)

    seen_pairs = set()
    for rule in result.rules:
        if len(rule.antecedents) == 1 and len(rule.consequents) == 1:
            key = frozenset(rule.antecedents) | frozenset(rule.consequents)
            assert key not in seen_pairs, f"symmetric duplicate rule for {key}"
            seen_pairs.add(key)


def test_analyze_baskets_truncates_and_reports_overflow(basket_transactions: pd.DataFrame):
    params = BasketParams(min_support=0.005, min_confidence=0.0, min_lift=0.5, max_rules=1)
    result = analyze_baskets(basket_transactions, params)

    assert result.total_rules_found > 1
    assert result.rules_truncated
    assert len(result.rules) == 1
    assert len(result.overflow_rules) == result.total_rules_found - 1
    assert any("top 1" in w for w in result.warnings)


def test_analyze_baskets_empty_result_has_warning(basket_transactions: pd.DataFrame):
    params = BasketParams(min_support=0.999, min_confidence=0.999, min_lift=100.0, max_rules=50)
    result = analyze_baskets(basket_transactions, params)

    assert result.rules == []
    assert result.total_rules_found == 0
    assert not result.rules_truncated
    assert any("No association rules" in w for w in result.warnings)


def test_analyze_baskets_with_day_of_week_dimension(basket_transactions: pd.DataFrame):
    params = BasketParams(
        min_support=0.02, min_confidence=0.1, min_lift=1.0, max_rules=50, dimension="day_of_week"
    )
    result = analyze_baskets(basket_transactions, params)

    assert result.dimension == "day_of_week"
    assert len(result.coverage) > 1
    assert all(c.slice_key is not None and c.slice_key.startswith("day_of_week=") for c in result.coverage)
    assert all(r.slice_key is not None for r in result.rules)
