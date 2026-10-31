import json
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

import pytest
from pydantic import ValidationError

from teammates.llm.client import ChatRequest, MeteredClient, TokenUsage
from teammates.llm.fake import FakeLLM
from teammates.orchestrator.core import Orchestrator
from teammates.orchestrator.eval import (
    EvalCase,
    Price,
    cost_inr,
    load_cases,
    main,
    run_eval,
    summarize,
    write_run,
)
from teammates.orchestrator.models import Party
from teammates.orchestrator.models import WorkflowType as W
from teammates.orchestrator.planner import LLMPlanner

DATA = Path(__file__).parents[2] / "data"
SMALL = DATA / "orchestrator_eval.json"
FULL = DATA / "orchestrator_eval_1000.json"
NOW = datetime(2026, 10, 30, tzinfo=UTC)


def _fake_clock(step_s: float = 0.002) -> Callable[[], float]:
    """Each call advances by step_s, so every case measures exactly step_s (single worker)."""
    ticks = count()
    return lambda: next(ticks) * step_s


def _case(i: int, query: str, expected: list[W]) -> EvalCase:
    return EvalCase(id=i, query=query, party=Party(type="merchant", id="M"), expected=expected)


def _answers(table: dict[str, list[str]]) -> Callable[[ChatRequest], str]:
    """Fake LLM that looks the query up in table; unknown queries get a non-JSON reply."""

    def reply(request: ChatRequest) -> str:
        query = json.loads(request.messages[-1].content)["query"]
        if query not in table:
            return "not json"
        return json.dumps({"workflows": table[query], "intent": "x"})

    return reply


def _orchestrator(
    table: dict[str, list[str]], usage: TokenUsage | None = None
) -> tuple[Orchestrator, MeteredClient]:
    client = MeteredClient(FakeLLM(_answers(table), usage=usage))
    return Orchestrator(LLMPlanner(client)), client


def test_small_fixture_scores_against_labels() -> None:
    cases = load_cases(SMALL)
    orch, _ = _orchestrator({c.query: [wf.value for wf in c.expected] for c in cases})
    assert all(r.correct for r in run_eval(orch, cases, clock=_fake_clock()))


def test_full_dataset_loads_with_1000_unique_cases() -> None:
    cases = load_cases(FULL)
    assert len(cases) == 1000
    assert len({c.query for c in cases}) == 1000


def test_cost_inr_bills_cached_tokens_at_cached_rate() -> None:
    price = Price(input=15, cached_input=5, output=60)
    usage = TokenUsage(prompt_tokens=1_000_000, cached_tokens=400_000, completion_tokens=100_000)
    assert cost_inr(usage, price) == pytest.approx(600_000 * 15e-6 + 400_000 * 5e-6 + 6)
    assert cost_inr(usage, None) == 0.0


def test_summary_counts_latency_labels_and_escalations() -> None:
    cases = [
        _case(1, "ticket", [W.SUPPORT]),
        _case(2, "ticket and loan", [W.SUPPORT]),  # model returns two workflows: wrong
        _case(3, "garbled", [W.COLLECTIONS]),  # model breaks the contract: escalated, wrong
    ]
    orch, _ = _orchestrator({"ticket": ["support"], "ticket and loan": ["support", "collections"]})
    results = run_eval(orch, cases, clock=_fake_clock(0.002))
    s = summarize(results, label="m", now=NOW)
    assert (s.total_queries, s.correct, s.wrong, s.accuracy_pct) == (3, 1, 2, 33.33)
    assert s.escalations == 1
    assert s.latency.avg_ms == 2.0
    assert s.latency.p95_ms == 2.0
    assert s.accuracy_by_label["support"].model_dump() == {
        "total": 2, "correct": 1, "accuracy_pct": 50.0
    }
    assert s.generated_at == "2026-10-30T00:00:00+00:00"


def test_run_meters_tokens_and_cost_per_case_in_parallel() -> None:
    usage = TokenUsage(prompt_tokens=1000, cached_tokens=0, completion_tokens=100)
    cases = [_case(i, f"q{i}", [W.SUPPORT]) for i in range(20)]
    orch, client = _orchestrator({c.query: ["support"] for c in cases}, usage=usage)
    price = Price(input=15, cached_input=5, output=60)
    results = run_eval(orch, cases, meter=client, price=price, workers=4)
    assert [r.id for r in results] == list(range(20))  # order preserved
    assert all(r.usage == usage for r in results)  # no cross-thread leakage
    s = summarize(results, label="m", price=price)
    assert s.tokens.prompt_tokens == 20_000
    assert s.cost_inr_total == pytest.approx(20 * (1000 * 15 + 100 * 60) / 1e6)
    assert s.escalations == 0


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_workers_rejected(workers: int) -> None:
    orch, _ = _orchestrator({})
    with pytest.raises(ValueError):
        run_eval(orch, [_case(1, "t", [W.SUPPORT])], workers=workers)


def test_empty_inputs_rejected() -> None:
    orch, _ = _orchestrator({})
    with pytest.raises(ValueError):
        run_eval(orch, [])
    with pytest.raises(ValueError):
        summarize([], label="x")


def test_load_cases_rejects_unknown_workflow(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": 1, "query": "q", "expected": ["nope"]}]))
    with pytest.raises(ValidationError):
        load_cases(bad)


def test_write_run_puts_wrong_cases_first(tmp_path: Path) -> None:
    cases = [_case(1, "ticket", [W.SUPPORT]), _case(2, "ticket", [W.KAM])]
    orch, _ = _orchestrator({"ticket": ["support"]})
    results = run_eval(orch, cases, clock=_fake_clock())
    write_run(summarize(results, label="m"), results, tmp_path / "out")
    details = json.loads((tmp_path / "out" / "m_details.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in details] == [2, 1]
    assert (tmp_path / "out" / "m_summary.json").exists()


def test_main_without_key_fails_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        main(["--cases", str(SMALL), "--out-dir", str(tmp_path)])


class _FakeWaitMeter:
    """Reports a fixed queue wait per case, like a MeteredRateLimiter would."""

    def __init__(self, wait_s: float) -> None:
        self.wait_s = wait_s

    def take(self) -> float:
        return self.wait_s


def test_queue_wait_is_reported_separately_from_latency() -> None:
    cases = [_case(1, "ticket", [W.SUPPORT]), _case(2, "ticket2", [W.SUPPORT])]
    orch, _ = _orchestrator({"ticket": ["support"], "ticket2": ["support"]})
    # Each case takes 45 ms end to end, 40 ms of which was rate-limit queueing.
    results = run_eval(orch, cases, clock=_fake_clock(0.045), wait_meter=_FakeWaitMeter(0.040))
    assert [(r.latency_ms, r.queue_wait_ms) for r in results] == [(5.0, 40.0), (5.0, 40.0)]
    s = summarize(results, label="m")
    assert s.latency.avg_ms == 5.0
    assert s.queue_wait.avg_ms == 40.0


def test_without_wait_meter_queue_wait_is_zero() -> None:
    orch, _ = _orchestrator({"ticket": ["support"]})
    [r] = run_eval(orch, [_case(1, "ticket", [W.SUPPORT])], clock=_fake_clock(0.003))
    assert (r.latency_ms, r.queue_wait_ms) == (3.0, 0.0)
