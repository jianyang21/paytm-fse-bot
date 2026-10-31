"""Sarvam AI chat-completions adapter (OpenAI-compatible endpoint), stdlib HTTP only."""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import SecretStr

from teammates.llm.client import ChatRequest, LLMError, LLMResponse, TokenUsage
from teammates.llm.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
# Sarvam reports some request errors from the model as HTTP 500 with this text,
# e.g. "Model call failed: Error code: 400 - ...". Retrying those can never succeed.
_WRAPPED_CLIENT_ERROR = re.compile(r"Error code: 4\d\d")


class Transport(Protocol):
    """Sends a POST and returns (status, body). Raises OSError on network failure."""

    def post(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> tuple[int, bytes]:
        """Perform the request."""
        ...


class UrllibTransport:
    """Default transport over urllib; non-2xx statuses are returned, not raised."""

    def post(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> tuple[int, bytes]:
        """POST body to url and return (status, response bytes)."""
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()


def is_retryable(status: int, body: str) -> bool:
    """Transient statuses are retried, except a 500 that wraps a model-side 4xx error."""
    if status not in RETRYABLE_STATUS:
        return False
    return not (status == 500 and _WRAPPED_CLIENT_ERROR.search(body))


def build_payload(request: ChatRequest) -> dict[str, Any]:
    """Translate a ChatRequest into the Sarvam/OpenAI request body."""
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
    }
    optional = {
        "seed": request.seed,
        "max_tokens": request.max_tokens,
        "reasoning_effort": request.reasoning_effort,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    if request.json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "schema": request.json_schema,
                "strict": True,
            },
        }
    return payload


def parse_response(data: dict[str, Any]) -> LLMResponse:
    """Extract content and usage; raise LLMError if there is no usable content."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Response has no choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMError("Empty completion content")
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return LLMResponse(
        content=content,
        model=str(data.get("model", "")),
        usage=TokenUsage(
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            cached_tokens=cached,
        ),
    )


class SarvamClient:
    """LLMClient for Sarvam. Rate-limited; retries timeouts, 429 and 5xx with backoff."""

    def __init__(
        self,
        api_key: SecretStr,
        rate_limiter: RateLimiter,
        base_url: str = "https://api.sarvam.ai/v1",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._transport = transport or UrllibTransport()
        self._sleep = sleep
        # Required: every Sarvam call is rate limited (Redis-backed in production).
        self._rate_limiter = rate_limiter

    def complete(self, request: ChatRequest) -> LLMResponse:
        """Send the request, retrying transient failures; raise LLMError when exhausted."""
        body = json.dumps(build_payload(request)).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self._api_key.get_secret_value(),
        }
        last_error = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(2 ** (attempt - 1))
            self._rate_limiter.acquire()  # every HTTP attempt, retries included, is counted
            try:
                status, raw = self._transport.post(self._url, headers, body, self._timeout_s)
            except OSError as exc:  # timeouts and connection errors
                last_error = f"network error: {type(exc).__name__}"
                logger.warning("sarvam attempt=%d %s", attempt + 1, last_error)
                continue
            if status == 200:
                return parse_response(self._decode(raw))
            snippet = raw[:200].decode("utf-8", "replace")
            last_error = f"HTTP {status}: {snippet}"
            logger.warning("sarvam attempt=%d HTTP %d: %s", attempt + 1, status, snippet)
            if not is_retryable(status, snippet):
                break
        raise LLMError(f"Sarvam request failed ({request.model}): {last_error}")

    @staticmethod
    def _decode(raw: bytes) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError("Sarvam returned non-JSON body") from exc
        if not isinstance(data, dict):
            raise LLMError("Sarvam returned a non-object JSON body")
        return data
