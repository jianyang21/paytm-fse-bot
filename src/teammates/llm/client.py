"""Provider-neutral chat interface plus a thread-safe token meter."""

import threading
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

ReasoningEffort = Literal["none", "low", "medium", "high", "max"]


class Message(BaseModel):
    """One chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Everything needed for one chat completion."""

    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    json_schema: dict[str, Any] | None = None
    schema_name: str = "response"


class TokenUsage(BaseModel):
    """Token counts for one or more calls. cached_tokens is a subset of prompt_tokens."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class LLMResponse(BaseModel):
    """The assistant's text plus usage for billing."""

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class LLMError(RuntimeError):
    """Any failure to get a usable completion (HTTP, timeout, empty reply)."""


class LLMClient(Protocol):
    """Anything that can complete a chat request."""

    def complete(self, request: ChatRequest) -> LLMResponse:
        """Return the completion or raise LLMError."""
        ...


class MeteredClient:
    """Wraps a client and accumulates token usage per thread, so concurrent callers
    can each read back only their own usage with take()."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._local = threading.local()

    def complete(self, request: ChatRequest) -> LLMResponse:
        """Delegate to the wrapped client and record its usage for this thread."""
        response = self._inner.complete(request)
        self._local.usage = self._current() + response.usage
        return response

    def take(self) -> TokenUsage:
        """Return usage recorded on this thread since the last take(), then reset it."""
        usage = self._current()
        self._local.usage = TokenUsage()
        return usage

    def _current(self) -> TokenUsage:
        return getattr(self._local, "usage", None) or TokenUsage()
