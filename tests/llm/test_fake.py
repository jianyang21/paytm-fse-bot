import pytest

from teammates.llm.client import ChatRequest, LLMError, Message, TokenUsage
from teammates.llm.fake import FakeLLM


def _req(text: str) -> ChatRequest:
    return ChatRequest(model="m", messages=[Message(role="user", content=text)])


def test_fake_replies_and_records() -> None:
    fake = FakeLLM(lambda r: r.messages[-1].content.upper(), usage=TokenUsage(prompt_tokens=3))
    res = fake.complete(_req("hi"))
    assert (res.content, res.model, res.usage.prompt_tokens) == ("HI", "m", 3)
    assert [r.messages[-1].content for r in fake.requests] == ["hi"]


def test_fake_can_raise() -> None:
    def boom(_: ChatRequest) -> str:
        raise LLMError("down")

    with pytest.raises(LLMError):
        FakeLLM(boom).complete(_req("x"))
