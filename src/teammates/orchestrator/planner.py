"""Planner: Sarvam LLM classifies intent; code fixes order and dependencies.

If the LLM call fails or its reply breaks the JSON contract, the request is escalated
to a human (the admin) instead of being routed.
"""

import logging
import uuid
from collections.abc import Callable
from itertools import count

from pydantic import ValidationError

from teammates.llm.client import ChatRequest, LLMClient, LLMError, ReasoningEffort
from teammates.orchestrator.models import (
    Escalation,
    ExecutionPlan,
    Party,
    RoutingMethod,
    WorkflowType,
)
from teammates.orchestrator.prompt import (
    OUTPUT_SCHEMA,
    SCHEMA_NAME,
    PlannerOutput,
    build_messages,
    parse_output,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sarvam-105b-conversations"  # A/B winner, see reports/orchestrator_eval/
ESCALATION_TARGET = "admin"  # Placeholder owner until the human reviewer role is decided.

# Fixed execution order; the copilot runs only when no other workflow applies.
EXECUTION_ORDER: tuple[WorkflowType, ...] = (
    WorkflowType.SUPPORT,
    WorkflowType.COLLECTIONS,
    WorkflowType.KAM,
    WorkflowType.FSE,
)

DEFAULT_ACTIONS: dict[WorkflowType, str] = {
    WorkflowType.SUPPORT: "resolve_ticket",
    WorkflowType.COLLECTIONS: "evaluate_loan_health",
    WorkflowType.KAM: "fetch_portfolio_metrics",
    WorkflowType.FSE: "check_fse_state",
    WorkflowType.COPILOT: "team_overview",
}

# Collections must not chase a merchant or customer before their open ticket is handled.
DEPENDENCIES: dict[WorkflowType, list[WorkflowType]] = {
    WorkflowType.COLLECTIONS: [WorkflowType.SUPPORT],
}

PlanIdFactory = Callable[[], str]


def sequential_plan_ids(prefix: str = "plan") -> PlanIdFactory:
    """Return a deterministic id factory: plan-000001, plan-000002, ..."""
    counter = count(1)
    return lambda: f"{prefix}-{next(counter):06d}"


def canonical_order(workflows: list[WorkflowType]) -> list[WorkflowType]:
    """Dedupe and sort into execution order; copilot only when nothing else applies."""
    order = [wf for wf in EXECUTION_ORDER if wf in set(workflows)]
    return order or [WorkflowType.COPILOT]


def _dependencies_within(order: list[WorkflowType]) -> dict[WorkflowType, list[WorkflowType]]:
    """Keep only dependencies whose prerequisite workflow is also in this plan."""
    deps: dict[WorkflowType, list[WorkflowType]] = {}
    for wf in order:
        present = [dep for dep in DEPENDENCIES.get(wf, []) if dep in order]
        if present:
            deps[wf] = present
    return deps


def build_plan(
    plan_id: str, workflows: list[WorkflowType], party: Party | None, intent: str | None = None
) -> ExecutionPlan:
    """Build a plan in canonical order with dependencies and default action payloads."""
    order = canonical_order(workflows)
    party_fields = {
        "party_type": party.type if party else None,
        "party_id": party.id if party else None,
    }
    return ExecutionPlan(
        plan_id=plan_id,
        party=party,
        intent=intent,
        execution_order=order,
        step_dependencies=_dependencies_within(order),
        action_payloads={wf: {"action": DEFAULT_ACTIONS[wf], **party_fields} for wf in order},
    )


def build_escalation(plan_id: str, party: Party | None, reason: str) -> ExecutionPlan:
    """A plan that runs no workflow and hands the request to a human."""
    return ExecutionPlan(
        plan_id=plan_id,
        party=party,
        escalation=Escalation(to=ESCALATION_TARGET, reason=reason),
    )


class LLMPlanner:
    """Asks the LLM which workflows apply; escalates to a human if that fails."""

    def __init__(
        self,
        client: LLMClient,
        model: str = DEFAULT_MODEL,
        *,
        seed: int = 42,
        max_tokens: int = 2048,
        reasoning_effort: ReasoningEffort | None = None,
        new_plan_id: PlanIdFactory | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._seed = seed
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._new_plan_id = new_plan_id or (lambda: str(uuid.uuid4()))

    def classify(self, query: str, party: Party | None) -> PlannerOutput:
        """Call the LLM and return its validated output. Raises LLMError or ValidationError."""
        request = ChatRequest(
            model=self._model,
            messages=build_messages(query, party),
            temperature=0.0,
            seed=self._seed,
            max_tokens=self._max_tokens,
            reasoning_effort=self._reasoning_effort,
            json_schema=OUTPUT_SCHEMA,
            schema_name=SCHEMA_NAME,
        )
        return parse_output(self._client.complete(request).content)

    def plan(self, query: str, party: Party | None) -> tuple[RoutingMethod, ExecutionPlan]:
        """Return ("llm", plan), or ("escalated", plan for a human) if the LLM step failed."""
        try:
            out = self.classify(query, party)
        except (LLMError, ValidationError) as exc:
            reason = f"LLM routing failed: {type(exc).__name__}"
            logger.warning("escalating to %s: %s", ESCALATION_TARGET, reason)
            return "escalated", build_escalation(self._new_plan_id(), party, reason)
        return "llm", build_plan(self._new_plan_id(), out.workflows, party, out.intent)
