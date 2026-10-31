"""Fixed, parameterized data tools for the support workflow, and the per-ticket cache.

The LLM never picks a tool or writes SQL: the resolver calls these by ticket type, through
call(), which allows only ALLOWED. MockTools serves the demo/seed merchants; a production
implementation has the same four methods over the real merchant, security and payments systems.

Cache: Redis, keyed by customer and open ticket (support:{merchant}:{ticket}:{part}), so a
repeat contact on the same ticket is answered without calling the tools again. Sensitive parts
get a short TTL (config.toml [cache]); invalidate() runs on every ticket state change. Redis
down -> the tools are called (never blocks a ticket).
"""

import json
import logging
from datetime import datetime, timedelta
from functools import cache
from importlib.resources import files
from typing import Any, Protocol

import redis

log = logging.getLogger(__name__)
ALLOWED = ("find_merchant", "get_user_details", "get_security_signals", "get_txn_status")
PARTS = ("profile", "security", "txns")


@cache
def seed() -> dict[str, Any]:
    """Demo merchants + labeled tickets (seed.json)."""
    return json.loads(files("teammates.support").joinpath("seed.json").read_text("utf-8"))


class Tools(Protocol):
    def find_merchant(self, contact: str) -> int | None: ...
    def get_user_details(self, merchant_id: int) -> dict[str, Any] | None: ...
    def get_security_signals(self, merchant_id: int) -> dict[str, Any]: ...
    def get_txn_status(self, merchant_id: int) -> list[dict[str, Any]]: ...


def call(tools: Tools, name: str, arg: int | str) -> Any:
    """The only way the workflow reaches data: allowlisted name, one typed parameter."""
    if name not in ALLOWED:
        raise PermissionError(f"tool {name!r} is not allowed for the support workflow")
    if (name == "find_merchant") != isinstance(arg, str) or isinstance(arg, bool):
        raise TypeError(f"bad parameter for {name}: {type(arg).__name__}")
    return getattr(tools, name)(arg)


class MockTools:
    """Seed merchants (seed.json "merchants"). Times in the seed are days before `now`."""

    def __init__(self, merchants: dict[str, dict[str, Any]], now: datetime) -> None:
        self.m = {int(k): v for k, v in merchants.items()}
        self.now = now
        self.calls: list[tuple[str, Any]] = []

    def _ago(self, days: float | None) -> str | None:
        return None if days is None else (self.now - timedelta(days=days)).isoformat()

    def find_merchant(self, contact: str) -> int | None:
        self.calls.append(("find_merchant", contact))
        c = contact.strip().lower()
        return next((mid for mid, m in self.m.items() if c in m["contacts"]), None)

    def get_user_details(self, merchant_id: int) -> dict[str, Any] | None:
        self.calls.append(("get_user_details", merchant_id))
        m = self.m.get(merchant_id)
        return m and {k: m[k] for k in ("name", "business", "tier", "kam_email", "language",
                                         "contacts")} | {"merchant_id": merchant_id}

    def get_security_signals(self, merchant_id: int) -> dict[str, Any]:
        self.calls.append(("get_security_signals", merchant_id))
        s = self.m[merchant_id]["security"]
        return {"two_fa": s["two_fa"], "sim_change_at": self._ago(s.get("sim_change_days_ago")),
                "failed_logins": [self._ago(d) for d in s.get("failed_logins_days_ago", [])],
                "blocked_now": s.get("blocked_now", False),
                "past_fraud": s.get("past_fraud", "none"), "identity": s.get("identity", "yes")}

    def get_txn_status(self, merchant_id: int) -> list[dict[str, Any]]:
        self.calls.append(("get_txn_status", merchant_id))
        return [t | {"date": self._ago(t["days_ago"])[:10]}
                for t in self.m[merchant_id].get("txns", [])]


def parse_signals(s: dict[str, Any]) -> dict[str, Any]:
    """Tool/cache JSON (ISO strings) -> what risk.likelihood expects (datetimes)."""
    at = datetime.fromisoformat
    return s | {"sim_change_at": s["sim_change_at"] and at(s["sim_change_at"]),
                "failed_logins": [at(t) for t in s["failed_logins"]]}


class ContextCache:
    def __init__(self, r: redis.Redis | None, ttls: dict[str, int]) -> None:
        self.r, self.ttls = r, ttls

    def key(self, merchant_id: int, ticket_id: int, part: str) -> str:
        return f"support:{merchant_id}:{ticket_id}:{part}"

    def get_or_fetch(self, merchant_id: int, ticket_id: int, part: str,
                     fetch: Any) -> tuple[Any, bool]:
        """(value, hit). fetch() runs on a miss and the result is cached for the part's TTL."""
        return self.cached(self.key(merchant_id, ticket_id, part), self.ttls[f"{part}_ttl_s"],
                           fetch)

    def cached(self, key: str, ttl_s: int, fetch: Any) -> tuple[Any, bool]:
        """Cache-then-source for any JSON value: (value, hit). Redis errors fall through to
        fetch() (logged), so a cache outage never blocks the caller. Shared by the support
        desk (per ticket) and the KAM digest (per KAM and day)."""
        try:
            raw = self.r.get(key) if self.r else None
        except redis.RedisError as exc:
            log.warning("cache get failed, calling the source: %s", exc)
            raw = None
        if raw is not None:
            return json.loads(raw), True
        value = fetch()
        try:
            if self.r:
                self.r.setex(key, ttl_s, json.dumps(value))
        except redis.RedisError as exc:
            log.warning("cache set failed: %s", exc)
        return value, False

    def invalidate(self, merchant_id: int, ticket_id: int) -> None:
        try:
            if self.r:
                self.r.delete(*(self.key(merchant_id, ticket_id, p) for p in PARTS))
        except redis.RedisError as exc:  # sensitive parts expire within security_ttl_s anyway
            log.warning("support cache invalidate failed: %s", exc)
