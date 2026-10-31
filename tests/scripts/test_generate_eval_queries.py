import json
import random
from collections import Counter
from pathlib import Path

import pytest

from scripts.generate_eval_queries import (
    CATEGORY_LABEL,
    EXECUTION_ORDER,
    PLAN,
    generate,
    main,
    make_case,
)


def test_generate_is_deterministic_per_seed() -> None:
    assert generate(42) == generate(42)
    assert generate(42) != generate(7)


def test_generate_counts_unique_queries_and_ids() -> None:
    cases = generate(42)
    assert len(cases) == sum(n for _, n in PLAN) == 1000
    assert [c["id"] for c in cases] == list(range(1, 1001))
    assert len({str(c["query"]).casefold() for c in cases}) == 1000
    assert Counter(c["category"] for c in cases) == dict(PLAN)


def test_labels_are_in_execution_order_and_match_category() -> None:
    for c in generate(42):
        expected = c["expected"]
        assert expected == [wf for wf in [*EXECUTION_ORDER, "copilot"] if wf in expected]
        category = str(c["category"])
        if category in CATEGORY_LABEL:
            assert expected == [CATEGORY_LABEL[category]]
        else:
            assert set(expected) == set(category.split("+"))


def test_party_is_merchant_customer_or_none() -> None:
    assert make_case(1, "copilot", random.Random(0))["party"] is None
    merchant = make_case(2, "support+collections", random.Random(0))["party"]
    assert merchant["type"] == "merchant" and merchant["id"].startswith("M_")
    for category in ("customer_support", "customer_collections"):
        customer = make_case(3, category, random.Random(0))["party"]
        assert customer["type"] == "customer" and customer["id"].startswith("C_")


def test_customer_cases_present_with_correct_labels() -> None:
    labels = {c["category"]: c["expected"] for c in generate(42)}
    assert labels["customer_support"] == ["support"]
    assert labels["customer_collections"] == ["collections"]


def test_committed_dataset_matches_generator() -> None:
    committed = Path(__file__).parents[2] / "data" / "orchestrator_eval_1000.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == generate(42)


def test_main_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "d" / "cases.json"
    main(["--seed", "3", "--out", str(out)])
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 1000


@pytest.mark.parametrize("category", ["support", "fse", "kam+fse"])
def test_make_case_query_has_no_outer_whitespace(category: str) -> None:
    query = str(make_case(1, category, random.Random(5))["query"])
    assert query.strip() == query
