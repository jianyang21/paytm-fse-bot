import pytest
from pydantic import ValidationError

from teammates.orchestrator.models import Escalation, ExecutionPlan, IntentRequest, WorkflowType


def test_intent_request_rejects_empty_query_and_user() -> None:
    with pytest.raises(ValidationError):
        IntentRequest(query="", user_id="u1")
    with pytest.raises(ValidationError):
        IntentRequest(query="hi", user_id="")


def test_plan_requires_workflows_or_escalation() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlan(plan_id="p", execution_order=[])


def test_plan_cannot_have_both_workflows_and_escalation() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlan(
            plan_id="p",
            execution_order=[WorkflowType.FSE],
            escalation=Escalation(to="admin", reason="r"),
        )


def test_escalation_only_plan_is_valid() -> None:
    plan = ExecutionPlan(plan_id="p", escalation=Escalation(to="admin", reason="r"))
    assert plan.execution_order == []


def test_escalation_needs_target_and_reason() -> None:
    with pytest.raises(ValidationError):
        Escalation(to="", reason="r")
    with pytest.raises(ValidationError):
        Escalation(to="admin", reason="")


def test_plan_defaults_are_not_shared_between_instances() -> None:
    a = ExecutionPlan(plan_id="a", execution_order=[WorkflowType.FSE])
    b = ExecutionPlan(plan_id="b", execution_order=[WorkflowType.FSE])
    a.action_payloads[WorkflowType.FSE] = {"action": "x"}
    assert b.action_payloads == {}


def test_workflow_parses_from_string_value() -> None:
    plan = ExecutionPlan(plan_id="p", execution_order=["support"])
    assert plan.execution_order == [WorkflowType.SUPPORT]
