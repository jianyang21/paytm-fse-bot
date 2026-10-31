"""Core orchestrator: plan -> deconflict -> guardrails."""

import logging
from typing import Protocol

from teammates.orchestrator.models import (
    ExecutionPlan,
    GuardrailStatus,
    IntentRequest,
    OrchestratorResponse,
    Party,
    RoutingMethod,
    WorkflowType,
)

logger = logging.getLogger(__name__)

# Workflows that act on one specific merchant or customer and cannot run without one.
# KAM is excluded: only a few large merchants have a key account manager.
PARTY_SCOPED: frozenset[WorkflowType] = frozenset(
    {WorkflowType.SUPPORT, WorkflowType.COLLECTIONS}
)
HIGH_RISK_ACTIONS: frozenset[str] = frozenset({"waive_fee"})
_SEVERITY: dict[GuardrailStatus, int] = {"passed": 0, "flagged": 1, "blocked": 2}


class Planner(Protocol):
    """Anything that turns a query into a plan (LLMPlanner in production, fakes in tests)."""

    def plan(self, query: str, party: Party | None) -> tuple[RoutingMethod, ExecutionPlan]:
        """Return the routing method used and the execution plan."""
        ...


def check_dependency_order(plan: ExecutionPlan) -> None:
    """Raise ValueError if any workflow is ordered before a prerequisite it depends on."""
    position = {wf: i for i, wf in enumerate(plan.execution_order)}
    for wf, deps in plan.step_dependencies.items():
        for dep in deps:
            if dep not in position or wf not in position or position[dep] > position[wf]:
                raise ValueError(f"{wf.value} must run after {dep.value} in plan {plan.plan_id}")


def deconflict(plan: ExecutionPlan) -> list[str]:
    """Enforce 'one party, one voice': validate ordering and note serialized outreach."""
    check_dependency_order(plan)
    if plan.party is None or len(plan.execution_order) < 2:
        return []
    sequence = " -> ".join(wf.value for wf in plan.execution_order)
    return [
        f"{plan.party.type.capitalize()} {plan.party.id}: workflows run one at a time "
        f"({sequence}) so the {plan.party.type} gets a single voice."
    ]


def apply_guardrails(plan: ExecutionPlan) -> tuple[GuardrailStatus, list[str]]:
    """Return the most severe guardrail status for the plan and the reasons for it."""
    findings: list[tuple[GuardrailStatus, str]] = []
    order = set(plan.execution_order)

    unscoped = sorted(wf.value for wf in order & PARTY_SCOPED)
    if unscoped and plan.party is None:
        findings.append(("blocked", f"No merchant or customer given for workflows: {unscoped}."))

    for wf, payload in plan.action_payloads.items():
        if payload.get("action") in HIGH_RISK_ACTIONS:
            findings.append(
                ("flagged", f"High-risk action '{payload['action']}' in {wf.value} "
                 "needs supervisor sign-off.")
            )

    if {WorkflowType.SUPPORT, WorkflowType.COLLECTIONS} <= order:
        findings.append(
            ("passed", "Open support ticket: collections outreach waits until support runs.")
        )

    status: GuardrailStatus = max(
        (s for s, _ in findings), key=_SEVERITY.__getitem__, default="passed"
    )
    return status, [msg for _, msg in findings]


class Orchestrator:
    """Coordinates a request: understand intent, plan, deconflict, apply guardrails."""

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    def coordinate(self, req: IntentRequest) -> OrchestratorResponse:
        """Build a checked plan for the request. Workflows are not executed here."""
        routing, plan = self._planner.plan(req.query, req.party)
        notes = deconflict(plan)
        status, warnings = apply_guardrails(plan)
        # The query is not logged: it may contain customer PII.
        logger.info(
            "plan=%s routing=%s workflows=%s guardrails=%s",
            plan.plan_id,
            routing,
            [wf.value for wf in plan.execution_order],
            status,
        )
        return OrchestratorResponse(
            intent=plan.intent or "unclassified",  # never the raw query: it may hold PII
            routing_method=routing,
            plan=plan,
            deconfliction_notes=notes,
            guardrail_status=status,
            guardrail_warnings=warnings,
        )
