"""In-memory LLM client for tests: no network, scripted replies."""

from collections.abc import Callable

from teammates.llm.client import ChatRequest, LLMResponse, TokenUsage


class FakeLLM:
    """Answers each request with reply(request); records every request it saw.

    reply may raise LLMError to simulate provider failures.
    """

    def __init__(
        self,
        reply: Callable[[ChatRequest], str],
        usage: TokenUsage | None = None,
    ) -> None:
        self._reply = reply
        self._usage = usage or TokenUsage()
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> LLMResponse:
        """Record the request and return the scripted reply."""
        self.requests.append(request)
        return LLMResponse(content=self._reply(request), model=request.model, usage=self._usage)
