import json

import pytest
from pydantic import SecretStr

from teammates.llm.adapters.sarvam import (
    SarvamClient,
    build_payload,
    is_retryable,
    parse_response,
)
from teammates.llm.client import ChatRequest, LLMError, Message
from teammates.llm.rate_limit import RateLimitExceeded

OK_BODY = {
    "model": "sarvam-105b",
    "choices": [{"message": {"content": '{"a": 1}'}}],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 60},
    },
}


class ScriptedTransport:
    """Returns (status, body) items in order; an Exception item is raised instead."""

    def __init__(self, *responses: tuple[int, dict | bytes] | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], dict]] = []

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_s: float):
        self.calls.append((url, headers, json.loads(body)))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, payload = item
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


class CountingLimiter:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self) -> float:
        self.calls += 1
        return 0.0


def _req(**kw: object) -> ChatRequest:
    return ChatRequest(model="sarvam-105b", messages=[Message(role="user", content="q")], **kw)


def _client(transport: ScriptedTransport, sleeps: list[float] | None = None) -> SarvamClient:
    record = sleeps if sleeps is not None else []
    return SarvamClient(
        SecretStr("sk_test"), CountingLimiter(), transport=transport, sleep=record.append
    )


def test_build_payload_minimal_omits_unset_fields() -> None:
    assert build_payload(_req()) == {
        "model": "sarvam-105b",
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0.0,
    }


def test_build_payload_with_schema_and_options() -> None:
    payload = build_payload(
        _req(seed=7, max_tokens=50, reasoning_effort="low", json_schema={"type": "object"},
             schema_name="s")
    )
    assert payload["seed"] == 7
    assert payload["max_tokens"] == 50
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "s", "schema": {"type": "object"}, "strict": True},
    }


def test_parse_response_reads_content_and_usage() -> None:
    res = parse_response(OK_BODY)
    assert res.content == '{"a": 1}'
    assert (res.usage.prompt_tokens, res.usage.completion_tokens, res.usage.cached_tokens) == (
        100, 20, 60
    )


def test_parse_response_tolerates_missing_usage() -> None:
    res = parse_response({"choices": [{"message": {"content": "x"}}]})
    assert res.usage.prompt_tokens == 0


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
     {"choices": [{"message": {"content": "  "}}]}],
)
def test_parse_response_rejects_unusable(body: dict) -> None:
    with pytest.raises(LLMError):
        parse_response(body)


def test_complete_sends_key_header_and_url() -> None:
    transport = ScriptedTransport((200, OK_BODY))
    _client(transport).complete(_req())
    url, headers, _ = transport.calls[0]
    assert url == "https://api.sarvam.ai/v1/chat/completions"
    assert headers["api-subscription-key"] == "sk_test"


def test_complete_retries_transient_errors_with_backoff() -> None:
    transport = ScriptedTransport((429, b"slow down"), TimeoutError(), (503, b""), (200, OK_BODY))
    sleeps: list[float] = []
    assert _client(transport, sleeps).complete(_req()).content == '{"a": 1}'
    assert sleeps == [1, 2, 4]


def test_complete_gives_up_after_max_retries() -> None:
    transport = ScriptedTransport(*[(500, b"err")] * 4)
    with pytest.raises(LLMError, match="HTTP 500"):
        _client(transport).complete(_req())
    assert len(transport.calls) == 4


def test_complete_does_not_retry_client_errors() -> None:
    transport = ScriptedTransport((401, b"bad key"))
    with pytest.raises(LLMError, match="HTTP 401"):
        _client(transport).complete(_req())
    assert len(transport.calls) == 1


def test_error_message_never_contains_the_key() -> None:
    transport = ScriptedTransport((400, b"bad request"))
    with pytest.raises(LLMError) as info:
        _client(transport).complete(_req())
    assert "sk_test" not in str(info.value)


def test_complete_rejects_non_json_body() -> None:
    with pytest.raises(LLMError, match="non-JSON"):
        _client(ScriptedTransport((200, b"<html>"))).complete(_req())


def test_every_attempt_including_retries_goes_through_the_limiter() -> None:
    limiter = CountingLimiter()
    transport = ScriptedTransport((503, b""), (200, OK_BODY))
    client = SarvamClient(SecretStr("sk_test"), limiter, transport=transport, sleep=lambda s: None)
    client.complete(_req())
    assert limiter.calls == 2


def test_rate_limit_rejection_stops_before_sending() -> None:
    class Refuse:
        def acquire(self) -> float:
            raise RateLimitExceeded("full")

    transport = ScriptedTransport((200, OK_BODY))
    client = SarvamClient(SecretStr("k"), Refuse(), transport=transport)
    with pytest.raises(LLMError):
        client.complete(_req())
    assert transport.calls == []


def test_client_cannot_be_built_without_a_rate_limiter() -> None:
    with pytest.raises(TypeError):
        SarvamClient(SecretStr("k"))  # type: ignore[call-arg]


WRAPPED_400 = (
    b'{"error":{"message":"Model call failed: Error code: 400 - {\'error\': '
    b'{\'message\': \'Grammar error\'}}","code":"model_call_failed"}}'
)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (429, "slow down", True),
        (503, "", True),
        (500, "internal error", True),
        (500, WRAPPED_400.decode(), False),
        (400, "bad request", False),
        (401, "bad key", False),
    ],
)
def test_is_retryable(status: int, body: str, expected: bool) -> None:
    assert is_retryable(status, body) is expected


def test_wrapped_client_error_is_not_retried() -> None:
    transport = ScriptedTransport((500, WRAPPED_400))
    sleeps: list[float] = []
    with pytest.raises(LLMError, match="Grammar error"):
        _client(transport, sleeps).complete(_req())
    assert len(transport.calls) == 1
    assert sleeps == []


def test_failed_attempt_logs_status_and_body(caplog: pytest.LogCaptureFixture) -> None:
    transport = ScriptedTransport((429, b"too many requests"), (200, OK_BODY))
    with caplog.at_level("WARNING"):
        _client(transport).complete(_req())
    assert "HTTP 429: too many requests" in caplog.text
    assert "sk_test" not in caplog.text
