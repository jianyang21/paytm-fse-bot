import logging

import pytest

from teammates.llm.fake import FakeLLM
from teammates.orchestrator.core import (
    Orchestrator,
    apply_guardrails,
    check_dependency_order,
    deconflict,
)
from teammates.orchestrator.models import (
    ExecutionPlan,
    IntentRequest,
    Party,
    RoutingMethod,
)
from teammates.orchestrator.models import WorkflowType as W
from teammates.orchestrator.planner import LLMPlanner

MERCHANT = Party(type="merchant", id="M_1")
CUSTOMER = Party(type="customer", id="C_1")


def _plan(order: list[W], party: Party | None = MERCHANT, **kwargs: object) -> ExecutionPlan:
    return ExecutionPlan(plan_id="p", party=party, execution_order=order, **kwargs)


class _FakePlanner:
    """Returns a fixed plan, standing in for the LLM planner."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self._plan = plan

    def plan(self, query: str, party: Party | None) -> tuple[RoutingMethod, ExecutionPlan]:
        return "llm", self._plan


# --- deconfliction ---------------------------------------------------------------


def test_deconflict_notes_multiple_workflows_on_one_merchant() -> None:
    notes = deconflict(
        _plan([W.SUPPORT, W.COLLECTIONS], step_dependencies={W.COLLECTIONS: [W.SUPPORT]})
    )
    assert notes == [
        "Merchant M_1: workflows run one at a time (support -> collections) "
        "so the merchant gets a single voice."
    ]


def test_deconflict_notes_one_voice_for_a_customer() -> None:
    notes = deconflict(
        _plan(
            [W.SUPPORT, W.COLLECTIONS],
            party=CUSTOMER,
            step_dependencies={W.COLLECTIONS: [W.SUPPORT]},
        )
    )
    assert notes == [
        "Customer C_1: workflows run one at a time (support -> collections) "
        "so the customer gets a single voice."
    ]


@pytest.mark.parametrize(
    "plan",
    [_plan([W.SUPPORT]), _plan([W.FSE, W.KAM], party=None)],
)
def test_deconflict_no_notes_when_nothing_to_serialize(plan: ExecutionPlan) -> None:
    assert deconflict(plan) == []


def test_dependency_ordered_wrongly_is_rejected() -> None:
    bad = _plan([W.COLLECTIONS, W.SUPPORT], step_dependencies={W.COLLECTIONS: [W.SUPPORT]})
    with pytest.raises(ValueError, match="collections must run after support"):
        check_dependency_order(bad)


def test_dependency_missing_from_plan_is_rejected() -> None:
    bad = _plan([W.COLLECTIONS], step_dependencies={W.COLLECTIONS: [W.SUPPORT]})
    with pytest.raises(ValueError):
        check_dependency_order(bad)


# --- guardrails ------------------------------------------------------------------


def test_guardrails_pass_for_simple_plan() -> None:
    assert apply_guardrails(_plan([W.SUPPORT])) == ("passed", [])


def test_guardrails_block_party_workflow_without_party() -> None:
    status, warnings = apply_guardrails(_plan([W.COLLECTIONS, W.FSE], party=None))
    assert status == "blocked"
    assert warnings == ["No merchant or customer given for workflows: ['collections']."]


@pytest.mark.parametrize("workflow", [W.FSE, W.COPILOT, W.KAM])
def test_guardrails_allow_non_party_scoped_workflows_without_party(workflow: W) -> None:
    assert apply_guardrails(_plan([workflow], party=None))[0] == "passed"


@pytest.mark.parametrize("workflow", [W.SUPPORT, W.COLLECTIONS])
def test_guardrails_pass_customer_support_and_loans(workflow: W) -> None:
    assert apply_guardrails(_plan([workflow], party=CUSTOMER)) == ("passed", [])


def test_guardrails_flag_high_risk_action() -> None:
    plan = _plan([W.COLLECTIONS], action_payloads={W.COLLECTIONS: {"action": "waive_fee"}})
    status, warnings = apply_guardrails(plan)
    assert status == "flagged"
    assert "supervisor sign-off" in warnings[0]


def test_guardrails_note_support_before_collections_without_flagging() -> None:
    status, warnings = apply_guardrails(_plan([W.SUPPORT, W.COLLECTIONS]))
    assert status == "passed"
    assert len(warnings) == 1


def test_guardrails_report_most_severe_status() -> None:
    plan = _plan(
        [W.COLLECTIONS],
        party=None,
        action_payloads={W.COLLECTIONS: {"action": "waive_fee"}},
    )
    status, warnings = apply_guardrails(plan)
    assert status == "blocked"
    assert len(warnings) == 2


# --- coordinate ------------------------------------------------------------------


def _llm_orchestrator(reply: str) -> Orchestrator:
    return Orchestrator(LLMPlanner(FakeLLM(lambda r: reply)))


def test_coordinate_end_to_end_multi_intent() -> None:
    query = "Resolve ticket #8812 and check collection"
    orch = _llm_orchestrator(
        '{"workflows": ["collections", "support"], "intent": "Resolve ticket, check loan"}'
    )
    res = orch.coordinate(IntentRequest(query=query, user_id="u", party=MERCHANT))
    assert res.routing_method == "llm"
    assert res.plan.execution_order == [W.SUPPORT, W.COLLECTIONS]
    assert res.guardrail_status == "passed"
    assert len(res.deconfliction_notes) == 1
    assert res.intent == "Resolve ticket, check loan"


def test_coordinate_escalates_to_admin_when_llm_fails() -> None:
    res = _llm_orchestrator("not json").coordinate(
        IntentRequest(query="q", user_id="u", party=CUSTOMER)
    )
    assert res.routing_method == "escalated"
    assert res.plan.execution_order == []
    assert res.plan.escalation is not None
    assert res.plan.escalation.to == "admin"
    assert res.intent == "unclassified"
    assert res.guardrail_status == "passed"
    assert res.deconfliction_notes == []


def test_coordinate_uses_injected_planner() -> None:
    planned = _plan([W.KAM])
    res = Orchestrator(planner=_FakePlanner(planned)).coordinate(
        IntentRequest(query="anything", user_id="u")
    )
    assert res.plan is planned


def test_coordinate_rejects_badly_ordered_plan_from_planner() -> None:
    bad = _plan([W.COLLECTIONS, W.SUPPORT], step_dependencies={W.COLLECTIONS: [W.SUPPORT]})
    with pytest.raises(ValueError):
        Orchestrator(planner=_FakePlanner(bad)).coordinate(IntentRequest(query="q", user_id="u"))


def test_coordinate_does_not_log_the_query(caplog: pytest.LogCaptureFixture) -> None:
    query = "ticket for merchant phone 9876543210"
    orch = _llm_orchestrator('{"workflows": ["support"], "intent": "x"}')
    with caplog.at_level(logging.INFO):
        orch.coordinate(IntentRequest(query=query, user_id="u", party=MERCHANT))
    assert caplog.records
    assert "9876543210" not in caplog.text
