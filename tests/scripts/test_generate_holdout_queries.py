import json
import re
from collections import Counter
from pathlib import Path

from scripts import generate_eval_queries as eval_set
from scripts.generate_holdout_queries import (
    AREAS,
    MERCHANTS,
    PLAN,
    TEMPLATES,
    generate,
    main,
)
from teammates.orchestrator.prompt import SYSTEM_PROMPT

ROOT = Path(__file__).parents[2]


def _prompt_example_queries() -> set[str]:
    return {q.casefold() for q in re.findall(r'"query": "([^"]+)"', SYSTEM_PROMPT)}


def test_generate_is_deterministic_per_seed() -> None:
    assert generate(7) == generate(7)
    assert generate(7) != generate(8)


def test_counts_labels_and_uniqueness() -> None:
    cases = generate(7)
    assert len(cases) == sum(n for _, n in PLAN) == 200
    assert [c["id"] for c in cases] == list(range(1, 201))
    assert len({str(c["query"]).casefold() for c in cases}) == 200
    assert Counter(c["category"] for c in cases) == dict(PLAN)
    assert Counter(c["expected"][0] for c in cases) == {"fse": 100, "copilot": 100}
    for c in cases:
        assert c["expected"] == [str(c["category"]).split("/")[0]]


def test_no_template_reused_from_eval_set() -> None:
    eval_templates = {t.casefold() for ts in eval_set.SINGLE.values() for t in ts}
    eval_templates |= {t.casefold() for ts in eval_set.CLAUSES.values() for t in ts}
    holdout_templates = {t.casefold() for ts in TEMPLATES.values() for t in ts}
    assert holdout_templates.isdisjoint(eval_templates)


def test_no_merchant_or_area_reused_from_eval_set() -> None:
    assert set(MERCHANTS).isdisjoint(eval_set.MERCHANTS)
    assert set(AREAS).isdisjoint(eval_set.AREAS)


def test_no_query_seen_in_eval_set_or_prompt_examples() -> None:
    eval_queries = {str(c["query"]).casefold() for c in eval_set.generate(42)}
    seen = eval_queries | _prompt_example_queries()
    assert all(str(c["query"]).casefold() not in seen for c in generate(7))


def test_party_only_when_merchant_named() -> None:
    for c in generate(7):
        named = any(m in str(c["query"]) for m in MERCHANTS)
        assert (c["party"] is not None) == named


def test_committed_holdout_matches_generator() -> None:
    committed = ROOT / "data" / "orchestrator_holdout_fse_copilot.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == generate(7)


def test_main_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "h.json"
    main(["--seed", "1", "--out", str(out)])
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 200
