"""Trace spans stored in Postgres (table trace_span), no external service needed.

One trace = one job run, one FSE inside a job, or one inbound message. Spans nest through a
context var; each records duration, ok/error, and for errors the exception and the file:line
where it was raised. `v_root_causes` (schema.sql) shows the deepest failed span of each trace.
"""

import logging
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    path: str  # "job.morning > fse > plan": where in the flow this step sits
    trace_id: str
    id: str
    parent_id: str | None
    fse_id: int | None
    started_at: datetime
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
    duration_ms: float = 0.0

    def fail(self, error: str) -> None:
        """Mark a handled failure (fallback used, HTTP error) without raising."""
        self.status, self.error = "error", error[:2000]


_current: ContextVar[Span | None] = ContextVar("span", default=None)
_sink: Callable[[Span], None] | None = None


def set_sink(fn: Callable[[Span], None] | None) -> None:
    """Where finished spans go (Store.record_span in the bot, None = log only)."""
    global _sink
    _sink = fn


def current() -> Span | None:
    return _current.get()


def describe(exc: BaseException) -> str:
    """'KeyError: 'x' at store.py:212 in plan': the exception and the frame that raised it."""
    frames = traceback.extract_tb(exc.__traceback__)
    where = f" at {Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name}" \
        if frames else ""
    return f"{type(exc).__name__}: {exc}{where}"


@contextmanager
def span(name: str, fse_id: int | None = None, trace_id: str | None = None,
         **attrs: Any) -> Iterator[Span]:
    """Time a step. Exceptions are recorded on the span and re-raised. fse_id is inherited
    from the parent span; trace_id joins an existing trace (e.g. a delivery callback)."""
    parent = _current.get()
    s = Span(
        name=name,
        path=f"{parent.path} > {name}" if parent else name,
        trace_id=trace_id or (parent.trace_id if parent else uuid.uuid4().hex),
        id=uuid.uuid4().hex[:16],
        parent_id=parent.id if parent else None,
        fse_id=fse_id if fse_id is not None else (parent.fse_id if parent else None),
        started_at=datetime.now(UTC),
        attrs=attrs,
    )
    token = _current.set(s)
    t0 = time.perf_counter()
    try:
        yield s
    except Exception as exc:
        s.fail(describe(exc))
        raise
    finally:
        s.duration_ms = (time.perf_counter() - t0) * 1000
        _current.reset(token)
        _emit(s)


def _emit(s: Span) -> None:
    if s.status == "error":
        log.warning("step failed: %s | %s", s.path, s.error)
    if _sink is None:
        return
    try:
        _sink(s)
    except Exception as exc:  # tracing must never break the bot (e.g. DB down)
        log.warning("span not stored (%s): %s %s", exc, s.path, s.status)


def mask(phone: str) -> str:
    """919876543210 -> 91******3210, for logs and traces."""
    return f"{phone[:2]}{'*' * (len(phone) - 6)}{phone[-4:]}" if len(phone) > 6 else "***"
