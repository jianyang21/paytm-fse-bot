"""Evaluation harness: routing accuracy, latency, tokens and cost over labeled queries.

Usage:
    python -m teammates.orchestrator.eval --limit 5   (smoke test)
    python -m teammates.orchestrator.eval             (full run, default model)
    python -m teammates.orchestrator.eval --models sarvam-105b sarvam-105b-conversations
"""

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import redis
from pydantic import BaseModel, TypeAdapter

from teammates.config import load_settings
from teammates.llm.adapters.sarvam import SarvamClient
from teammates.llm.client import MeteredClient, ReasoningEffort, TokenUsage
from teammates.llm.rate_limit import MeteredRateLimiter, RedisSlidingWindowRateLimiter
from teammates.orchestrator.core import Orchestrator
from teammates.orchestrator.models import (
    GuardrailStatus,
    IntentRequest,
    Party,
    RoutingMethod,
    WorkflowType,
)
from teammates.orchestrator.planner import DEFAULT_MODEL, LLMPlanner
from teammates.orchestrator.prompt import PROMPT_SHA256, PROMPT_VERSION

logger = logging.getLogger(__name__)


class Price(BaseModel):
    """INR per 1M tokens, excluding GST."""

    input: float
    cached_input: float
    output: float


# Source: docs.sarvam.ai/api-reference-docs/pricing, checked 2026-10-30.
SARVAM_PRICES_INR: dict[str, Price] = {
    "sarvam-105b": Price(input=15, cached_input=5, output=60),
    "sarvam-105b-conversations": Price(input=15, cached_input=5, output=60),
}


class Meter(Protocol):
    """Reports the token usage the current thread spent since the last take()."""

    def take(self) -> TokenUsage:
        """Return and reset this thread's usage."""
        ...


class WaitMeter(Protocol):
    """Reports seconds the current thread spent queued for a rate-limit slot."""

    def take(self) -> float:
        """Return and reset this thread's wait."""
        ...


class EvalCase(BaseModel):
    """One labeled query with the workflows it should be routed to, in execution order."""

    id: int
    query: str
    party: Party | None = None
    expected: list[WorkflowType]


class CaseResult(BaseModel):
    """Outcome of running one EvalCase through the orchestrator."""

    id: int
    query: str
    routing: RoutingMethod
    execution_order: list[WorkflowType]
    expected: list[WorkflowType]
    correct: bool
    latency_ms: float  # end-to-end time minus rate-limit queueing
    queue_wait_ms: float  # time blocked waiting for a rate-limit slot
    guardrail_status: GuardrailStatus
    usage: TokenUsage
    cost_inr: float


class LatencyStats(BaseModel):
    """Per-query latency distribution in milliseconds."""

    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class LabelAccuracy(BaseModel):
    """Accuracy for one expected label, e.g. "support+collections"."""

    total: int
    correct: int
    accuracy_pct: float


class EvalSummary(BaseModel):
    """Headline metrics for one planner/model run."""

    label: str
    model: str | None
    prompt_version: str | None
    prompt_sha256: str | None
    generated_at: str
    total_queries: int
    correct: int
    wrong: int
    accuracy_pct: float
    escalations: int
    latency: LatencyStats  # provider + code time, excluding rate-limit queueing
    queue_wait: LatencyStats  # rate-limit queueing per query
    tokens: TokenUsage
    cost_inr_total: float
    cost_inr_per_query: float
    price_inr_per_1m_tokens: Price | None
    accuracy_by_label: dict[str, LabelAccuracy]


def load_cases(path: Path) -> list[EvalCase]:
    """Load and validate labeled eval cases from a JSON file."""
    return TypeAdapter(list[EvalCase]).validate_json(path.read_bytes())


def cost_inr(usage: TokenUsage, price: Price | None) -> float:
    """Cost of usage in INR; cached prompt tokens are billed at the cached rate."""
    if price is None:
        return 0.0
    fresh = usage.prompt_tokens - usage.cached_tokens
    return (
        fresh * price.input
        + usage.cached_tokens * price.cached_input
        + usage.completion_tokens * price.output
    ) / 1_000_000


def run_case(
    orchestrator: Orchestrator,
    case: EvalCase,
    clock: Callable[[], float],
    meter: Meter | None,
    price: Price | None,
    wait_meter: WaitMeter | None = None,
) -> CaseResult:
    """Run one case: time it, split out rate-limit queueing, charge its metered tokens."""
    for m in (meter, wait_meter):
        if m is not None:
            m.take()  # drop anything left over on this worker thread
    start = clock()
    res = orchestrator.coordinate(
        IntentRequest(query=case.query, party=case.party, user_id="eval_runner")
    )
    total_ms = (clock() - start) * 1000
    usage = meter.take() if meter is not None else TokenUsage()
    queue_ms = wait_meter.take() * 1000 if wait_meter is not None else 0.0
    return CaseResult(
        id=case.id,
        query=case.query,
        routing=res.routing_method,
        execution_order=res.plan.execution_order,
        expected=case.expected,
        correct=res.plan.execution_order == case.expected,
        latency_ms=round(max(0.0, total_ms - queue_ms), 3),
        queue_wait_ms=round(queue_ms, 3),
        guardrail_status=res.guardrail_status,
        usage=usage,
        cost_inr=cost_inr(usage, price),
    )


def run_eval(
    orchestrator: Orchestrator,
    cases: list[EvalCase],
    *,
    clock: Callable[[], float] = time.perf_counter,
    meter: Meter | None = None,
    wait_meter: WaitMeter | None = None,
    price: Price | None = None,
    workers: int = 1,
) -> list[CaseResult]:
    """Run every case (in parallel when workers > 1) and return results in case order."""
    if not cases:
        raise ValueError("No eval cases given.")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    def one(case: EvalCase) -> CaseResult:
        return run_case(orchestrator, case, clock, meter, price, wait_meter)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, cases))


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an ascending list."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def latency_stats(values_ms: list[float]) -> LatencyStats:
    """Total, mean and nearest-rank p50/p95/max of a non-empty list of milliseconds."""
    ordered = sorted(values_ms)
    return LatencyStats(
        total_ms=round(sum(ordered), 3),
        avg_ms=round(sum(ordered) / len(ordered), 3),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        max_ms=ordered[-1],
    )


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2)


def summarize(
    results: list[CaseResult],
    *,
    label: str,
    model: str | None = None,
    prompt_version: str | None = None,
    prompt_sha256: str | None = None,
    price: Price | None = None,
    now: datetime | None = None,
) -> EvalSummary:
    """Aggregate per-case results into headline metrics."""
    if not results:
        raise ValueError("No results to summarize.")
    n = len(results)
    correct = sum(r.correct for r in results)
    tokens = sum((r.usage for r in results), TokenUsage())
    total_cost = sum(r.cost_inr for r in results)

    by_label: dict[str, list[bool]] = {}
    for r in results:
        by_label.setdefault("+".join(wf.value for wf in r.expected), []).append(r.correct)

    return EvalSummary(
        label=label,
        model=model,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        generated_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
        total_queries=n,
        correct=correct,
        wrong=n - correct,
        accuracy_pct=_pct(correct, n),
        escalations=sum(r.routing == "escalated" for r in results),
        latency=latency_stats([r.latency_ms for r in results]),
        queue_wait=latency_stats([r.queue_wait_ms for r in results]),
        tokens=tokens,
        cost_inr_total=round(total_cost, 6),
        cost_inr_per_query=round(total_cost / n, 8),
        price_inr_per_1m_tokens=price,
        accuracy_by_label={
            key: LabelAccuracy(total=len(v), correct=sum(v), accuracy_pct=_pct(sum(v), len(v)))
            for key, v in sorted(by_label.items())
        },
    )


def write_json(data: BaseModel | list[BaseModel], path: Path) -> None:
    """Write a model (or list of models) as indented JSON, creating parent folders."""
    payload = [d.model_dump(mode="json") for d in data] if isinstance(data, list) else (
        data.model_dump(mode="json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_run(summary: EvalSummary, results: list[CaseResult], out_dir: Path) -> None:
    """Write <label>_summary.json and <label>_details.json (wrong cases first)."""
    write_json(summary, out_dir / f"{summary.label}_summary.json")
    ordered = sorted(results, key=lambda r: (r.correct, r.id))
    write_json(ordered, out_dir / f"{summary.label}_details.json")


def _evaluate_sarvam(
    cases: list[EvalCase], model: str, workers: int, reasoning_effort: ReasoningEffort | None
) -> tuple[EvalSummary, list[CaseResult]]:
    settings = load_settings()
    redis_client = redis.Redis.from_url(settings.redis_url.get_secret_value())
    redis_client.ping()  # fail fast: without Redis every call would escalate
    limiter = MeteredRateLimiter(RedisSlidingWindowRateLimiter(redis_client))
    client = MeteredClient(
        SarvamClient(settings.sarvam_api_key, limiter, base_url=settings.sarvam_base_url)
    )
    planner = LLMPlanner(client, model, reasoning_effort=reasoning_effort)
    price = SARVAM_PRICES_INR.get(model)
    if price is None:
        logger.warning("No price known for %s; cost will be reported as 0", model)
    results = run_eval(
        Orchestrator(planner),
        cases,
        meter=client,
        wait_meter=limiter,
        price=price,
        workers=workers,
    )
    summary = summarize(
        results,
        label=model,
        model=model,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=PROMPT_SHA256,
        price=price,
    )
    return summary, results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; writes one summary and one details file per run, plus comparison.json."""
    parser = argparse.ArgumentParser(description="Orchestrator routing eval")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--cases", type=Path, default=Path("data/orchestrator_eval_1000.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/orchestrator_eval"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N cases")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "max"])
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)[: args.limit]
    runs = [_evaluate_sarvam(cases, m, args.workers, args.reasoning_effort) for m in args.models]

    for summary, results in runs:
        write_run(summary, results, args.out_dir)
        logger.info(
            "%s: %d/%d correct (%.2f%%), latency avg %.0f ms p95 %.0f ms, "
            "queue wait avg %.0f ms, cost INR %.4f",
            summary.label,
            summary.correct,
            summary.total_queries,
            summary.accuracy_pct,
            summary.latency.avg_ms,
            summary.latency.p95_ms,
            summary.queue_wait.avg_ms,
            summary.cost_inr_total,
        )
    write_json([s for s, _ in runs], args.out_dir / "comparison.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
