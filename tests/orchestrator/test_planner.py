import json
import logging

import pytest

from teammates.llm.client import ChatRequest, LLMError
from teammates.llm.fake import FakeLLM
from teammates.orchestrator.models import Party
from teammates.orchestrator.models import WorkflowType as W
from teammates.orchestrator.planner import (
    DEFAULT_MODEL,
    ESCALATION_TARGET,
    LLMPlanner,
    build_escalation,
    build_plan,
    canonical_order,
    sequential_plan_ids,
)
from teammates.orchestrator.prompt import OUTPUT_SCHEMA, SCHEMA_NAME

MERCHANT = Party(type="merchant", id="M_1")
CUSTOMER = Party(type="customer", id="C_9")


def _reply(workflows: list[str]) -> str:
    return json.dumps({"workflows": workflows, "intent": "x"})


def _planner(reply: str | None = None, **kwargs: object) -> tuple[LLMPlanner, FakeLLM]:
    fake = FakeLLM(lambda r: reply if reply is not None else _reply(["fse"]))
    return LLMPlanner(fake, **kwargs), fake


# --- pure plan building ---------------------------------------------------------


def test_sequential_plan_ids_are_deterministic_and_independent() -> None:
    a, b = sequential_plan_ids(), sequential_plan_ids("x")
    assert [a(), a(), b()] == ["plan-000001", "plan-000002", "x-000001"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ([W.FSE, W.SUPPORT, W.FSE], [W.SUPPORT, W.FSE]),
        ([W.COLLECTIONS, W.SUPPORT], [W.SUPPORT, W.COLLECTIONS]),
        ([W.KAM, W.COPILOT], [W.KAM]),
        ([W.COPILOT, W.COPILOT], [W.COPILOT]),
        ([], [W.COPILOT]),
    ],
)
def test_canonical_order(given: list[W], expected: list[W]) -> None:
    assert canonical_order(given) == expected


def test_build_plan_sets_order_dependencies_and_payloads() -> None:
    plan = build_plan("p1", [W.COLLECTIONS, W.SUPPORT], CUSTOMER)
    assert plan.plan_id == "p1"
    assert plan.execution_order == [W.SUPPORT, W.COLLECTIONS]
    assert plan.step_dependencies == {W.COLLECTIONS: [W.SUPPORT]}
    assert plan.action_payloads[W.COLLECTIONS] == {
        "action": "evaluate_loan_health", "party_type": "customer", "party_id": "C_9"
    }
    assert plan.escalation is None


def test_collections_alone_has_no_dangling_dependency() -> None:
    assert build_plan("p", [W.COLLECTIONS], MERCHANT).step_dependencies == {}


def test_build_plan_without_party_has_null_party_fields() -> None:
    plan = build_plan("p2", [W.FSE], None)
    assert plan.party is None
    assert plan.action_payloads[W.FSE] == {
        "action": "check_fse_state", "party_type": None, "party_id": None
    }


def test_build_escalation_runs_no_workflow() -> None:
    plan = build_escalation("p3", MERCHANT, "why")
    assert plan.execution_order == []
    assert plan.escalation is not None
    assert (plan.escalation.to, plan.escalation.reason) == (ESCALATION_TARGET, "why")
    assert plan.party == MERCHANT


# --- LLM planner ----------------------------------------------------------------


def test_default_model_is_the_ab_winner() -> None:
    planner, fake = _planner()
    planner.plan("q", None)
    assert DEFAULT_MODEL == "sarvam-105b-conversations"
    assert fake.requests[0].model == DEFAULT_MODEL


def test_llm_plan_uses_model_output_in_canonical_order() -> None:
    planner, _ = _planner(_reply(["collections", "support"]))
    routing, plan = planner.plan("q", MERCHANT)
    assert routing == "llm"
    assert plan.execution_order == [W.SUPPORT, W.COLLECTIONS]
    assert plan.step_dependencies == {W.COLLECTIONS: [W.SUPPORT]}


def test_copilot_mixed_with_real_work_is_dropped() -> None:
    planner, _ = _planner(_reply(["copilot", "fse"]))
    assert planner.plan("q", None)[1].execution_order == [W.FSE]


def test_request_is_deterministic_and_schema_constrained() -> None:
    planner, fake = _planner(model="sarvam-105b", seed=7, reasoning_effort="low")
    planner.plan("q", CUSTOMER)
    req: ChatRequest = fake.requests[0]
    assert (req.model, req.temperature, req.seed, req.reasoning_effort) == (
        "sarvam-105b", 0.0, 7, "low"
    )
    assert req.json_schema == OUTPUT_SCHEMA
    assert req.schema_name == SCHEMA_NAME
    assert json.loads(req.messages[1].content)["party"] == {"type": "customer", "id": "C_9"}


@pytest.mark.parametrize(
    "bad_reply", ["sorry, I can't", '{"workflows": ["billing"], "intent": "x"}']
)
def test_contract_violation_escalates_to_admin(bad_reply: str) -> None:
    planner, _ = _planner(bad_reply)
    routing, plan = planner.plan("Resolve support ticket #1", MERCHANT)
    assert routing == "escalated"
    assert plan.execution_order == []
    assert plan.escalation is not None
    assert plan.escalation.to == "admin"
    assert plan.escalation.reason == "LLM routing failed: ValidationError"


def test_provider_error_escalates_to_admin(caplog: pytest.LogCaptureFixture) -> None:
    def down(_: ChatRequest) -> str:
        raise LLMError("503 from provider")

    with caplog.at_level(logging.WARNING):
        routing, plan = LLMPlanner(FakeLLM(down)).plan("my phone 9876543210", CUSTOMER)
    assert routing == "escalated"
    assert plan.escalation is not None
    assert plan.escalation.reason == "LLM routing failed: LLMError"
    assert "9876543210" not in caplog.text


def test_plan_ids_continue_across_escalations() -> None:
    replies = iter([_reply(["fse"]), "bad", _reply(["fse"])])
    planner = LLMPlanner(FakeLLM(lambda r: next(replies)), new_plan_id=sequential_plan_ids())
    ids = [planner.plan("task", None)[1].plan_id for _ in range(3)]
    assert ids == ["plan-000001", "plan-000002", "plan-000003"]


def test_default_plan_ids_are_unique_across_planners() -> None:
    a, _ = _planner()
    b, _ = _planner()
    assert a.plan("q", None)[1].plan_id != b.plan("q", None)[1].plan_id


def test_injected_id_factory_is_used() -> None:
    planner, _ = _planner(new_plan_id=lambda: "fixed")
    assert planner.plan("q", None)[1].plan_id == "fixed"
