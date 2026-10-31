import threading

import pytest
from pydantic import ValidationError

from teammates.llm.client import ChatRequest, Message, MeteredClient, TokenUsage
from teammates.llm.fake import FakeLLM


def _req() -> ChatRequest:
    return ChatRequest(model="m", messages=[Message(role="user", content="hi")])


def test_chat_request_needs_a_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[])


def test_token_usage_adds() -> None:
    total = TokenUsage(prompt_tokens=10, completion_tokens=2, cached_tokens=4) + TokenUsage(
        prompt_tokens=5, completion_tokens=1
    )
    assert total == TokenUsage(prompt_tokens=15, completion_tokens=3, cached_tokens=4)


def test_metered_client_accumulates_and_resets() -> None:
    client = MeteredClient(FakeLLM(lambda r: "ok", usage=TokenUsage(prompt_tokens=7)))
    client.complete(_req())
    client.complete(_req())
    assert client.take().prompt_tokens == 14
    assert client.take() == TokenUsage()


def test_metered_client_isolates_threads() -> None:
    client = MeteredClient(FakeLLM(lambda r: "ok", usage=TokenUsage(prompt_tokens=1)))
    seen: dict[str, int] = {}

    def work(name: str, calls: int) -> None:
        for _ in range(calls):
            client.complete(_req())
        seen[name] = client.take().prompt_tokens

    threads = [threading.Thread(target=work, args=(f"t{i}", i + 1)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"t0": 1, "t1": 2, "t2": 3, "t3": 4}
    assert client.take() == TokenUsage()  # main thread spent nothing
