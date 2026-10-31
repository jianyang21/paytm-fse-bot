"""Pydantic models that cross the orchestrator boundary."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class WorkflowType(StrEnum):
    """The five teammate workflows the orchestrator can run."""

    SUPPORT = "support"
    COLLECTIONS = "collections"
    KAM = "kam"
    FSE = "fse"
    COPILOT = "copilot"


RoutingMethod = Literal["llm", "escalated"]
GuardrailStatus = Literal["passed", "flagged", "blocked"]
PartyType = Literal["merchant", "customer"]


class Party(BaseModel):
    """Who a request is about: a merchant (business) or a customer (individual)."""

    type: PartyType
    id: str = Field(min_length=1)


class IntentRequest(BaseModel):
    """A natural-language request from a user, optionally about one merchant or customer."""

    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    party: Party | None = None


class Escalation(BaseModel):
    """Hand-off of a request to a human because it could not be routed automatically."""

    to: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExecutionPlan(BaseModel):
    """Ordered workflows to run, or an escalation to a human; never both, never neither."""

    plan_id: str
    party: Party | None = None
    intent: str | None = None  # LLM's PII-free summary; None when escalated
    execution_order: list[WorkflowType] = Field(default_factory=list)
    step_dependencies: dict[WorkflowType, list[WorkflowType]] = Field(default_factory=dict)
    action_payloads: dict[WorkflowType, dict[str, Any]] = Field(default_factory=dict)
    escalation: Escalation | None = None

    @model_validator(mode="after")
    def _workflows_xor_escalation(self) -> "ExecutionPlan":
        if bool(self.execution_order) == (self.escalation is not None):
            raise ValueError("A plan needs either workflows or an escalation, not both")
        return self


class OrchestratorResponse(BaseModel):
    """Result of coordinating one request: the plan plus deconfliction and guardrail output."""

    intent: str
    routing_method: RoutingMethod
    plan: ExecutionPlan
    deconfliction_notes: list[str] = Field(default_factory=list)
    guardrail_status: GuardrailStatus
    guardrail_warnings: list[str] = Field(default_factory=list)
