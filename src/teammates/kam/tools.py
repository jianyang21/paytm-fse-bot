"""Fixed, parameterized, typed tools for the KAM digest. The LLM never picks a tool or writes
SQL: the digest calls these through call(), which allows only REGISTRY's names, validates the
arguments against the input model and returns only the output model's declared fields.

    get_key_accounts(kam_id)            the KAM's assigned key accounts (merchant directory)
    get_txn_status(merchant_id)         recent transactions (the support desk's tool, reused)
    get_meetings(kam_id, day)           the KAM's calendar for one local day
    get_competitor_updates(since)       updates from the fixed competitor sources

kam_id is the KAM's login email (support_reviewer.email), the key the support desk assigns
key-account tickets with. MockKamTools serves the seed data (tests, evals); PgKamTools reads
the kam_meeting and competitor_update tables (schema.sql) and is what the bot runs.
"""

import json
import logging
from datetime import date, datetime, time, timedelta
from functools import cache
from importlib.resources import files
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, StrictInt, field_validator

from teammates.guard import EMAIL_RE
from teammates.support import tools as support_tools

log = logging.getLogger(__name__)


@cache
def seed() -> dict[str, Any]:
    """Demo KAMs, extra key accounts, meetings and the mocked competitor feed (seed.json)."""
    return json.loads(files("teammates.kam").joinpath("seed.json").read_text("utf-8"))


class UnknownKam(LookupError):
    """kam_id is not an active KAM."""


# --- inputs: extra arguments are an error ----------------------------------------------------

class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KamIn(_In):
    kam_id: str

    @field_validator("kam_id")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.fullmatch(v):
            raise ValueError("kam_id must be the KAM's email")
        return v


class MeetingsIn(KamIn):
    day: date


class UpdatesIn(_In):
    since: AwareDatetime


class MerchantIn(_In):
    merchant_id: StrictInt


# --- outputs: undeclared fields are dropped ---------------------------------------------------

class Account(BaseModel):
    merchant_id: int
    business: str
    name: str
    language: str


class Txn(BaseModel):
    ref: str
    kind: str
    amount: int
    status: str
    date: str


class Meeting(BaseModel):
    starts_at: AwareDatetime
    ends_at: AwareDatetime
    title: str
    merchant_id: int | None = None
    location: str | None = None


class CompetitorUpdate(BaseModel):
    competitor: str
    source: str
    title: str
    body: str
    url: str
    published_at: AwareDatetime

    @field_validator("url")
    @classmethod
    def _https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("every update needs its https source link")
        return v


REGISTRY: dict[str, tuple[type[_In], type[BaseModel]]] = {
    "get_key_accounts": (KamIn, Account),
    "get_txn_status": (MerchantIn, Txn),
    "get_meetings": (MeetingsIn, Meeting),
    "get_competitor_updates": (UpdatesIn, CompetitorUpdate),
}
ALLOWED = tuple(REGISTRY)


class KamTools(Protocol):
    merchants: dict[int, dict[str, Any]]

    def get_key_accounts(self, kam_id: str) -> list[dict[str, Any]]: ...
    def get_txn_status(self, merchant_id: int) -> list[dict[str, Any]]: ...
    def get_meetings(self, kam_id: str, day: date) -> list[dict[str, Any]]: ...
    def get_competitor_updates(self, since: datetime) -> list[dict[str, Any]]: ...


def call(tools: KamTools, name: str, **kwargs: Any) -> list[Any]:
    """The only way the KAM digest reaches data. PermissionError for a tool outside the
    allowlist, pydantic ValidationError (a ValueError) for bad arguments."""
    if name not in REGISTRY:
        raise PermissionError(f"tool {name!r} is not allowed for the KAM workflow")
    in_model, out_model = REGISTRY[name]
    args = in_model.model_validate(kwargs)
    return [out_model.model_validate(r) for r in getattr(tools, name)(**dict(args))]


def fixed_source_items(feed: list[dict[str, Any]], sources: list[Any],
                       now: datetime) -> list[dict[str, Any]]:
    """The feed fetch: only items whose link is under one of the fixed sources, dated."""
    out = []
    for item in feed:
        src = next((s for s in sources if item["url"].startswith(s.url)), None)
        if src is None:
            log.warning("competitor item dropped, not a fixed source: %s", item["url"])
            continue
        out.append(item | {"source": src.name, "competitor": src.competitor,
                           "published_at": now - timedelta(days=item["published_days_ago"])})
    return out


class _Directory:
    """Key-account assignment and transactions, from the merchant directory (the support
    seed in this build; the real merchant system later, same methods)."""

    def __init__(self, merchants: dict[str, dict[str, Any]], now: datetime) -> None:
        self.merchants = {int(k): v for k, v in merchants.items()}
        self.support = support_tools.MockTools(merchants, now)
        self.calls: list[tuple[str, Any]] = []

    def kams(self) -> set[str]:
        raise NotImplementedError

    def _known(self, kam_id: str) -> None:
        if kam_id not in self.kams():
            raise UnknownKam(f"no active KAM {kam_id}")

    def get_key_accounts(self, kam_id: str) -> list[dict[str, Any]]:
        self.calls.append(("get_key_accounts", kam_id))
        self._known(kam_id)
        return [m | {"merchant_id": mid} for mid, m in sorted(self.merchants.items())
                if m["tier"] == "key" and (m.get("kam_email") or "").lower() == kam_id]

    def get_txn_status(self, merchant_id: int) -> list[dict[str, Any]]:
        self.calls.append(("get_txn_status", merchant_id))
        return support_tools.call(self.support, "get_txn_status", merchant_id)


class MockKamTools(_Directory):
    """Seed data in memory. Meetings are placed on the run day (`now` in tz)."""

    def __init__(self, merchants: dict[str, dict[str, Any]], kams: list[str],
                 meetings: list[dict[str, Any]], feed: list[dict[str, Any]],
                 sources: list[Any], now: datetime, tz: ZoneInfo) -> None:
        super().__init__(merchants, now)
        self._kams, self.tz = {k.lower() for k in kams}, tz
        today = now.astimezone(tz).date()
        self.meetings = [_meeting_row(m, today, tz) for m in meetings]
        self.feed = fixed_source_items(feed, sources, now)

    def kams(self) -> set[str]:
        return self._kams

    def get_meetings(self, kam_id: str, day: date) -> list[dict[str, Any]]:
        self.calls.append(("get_meetings", (kam_id, day)))
        self._known(kam_id)
        return sorted((m for m in self.meetings if m["kam_id"] == kam_id
                       and m["starts_at"].astimezone(self.tz).date() == day),
                      key=lambda m: m["starts_at"])

    def get_competitor_updates(self, since: datetime) -> list[dict[str, Any]]:
        self.calls.append(("get_competitor_updates", since))
        return sorted((i for i in self.feed if i["published_at"] >= since),
                      key=lambda i: i["published_at"], reverse=True)


def _meeting_row(m: dict[str, Any], today: date, tz: ZoneInfo) -> dict[str, Any]:
    day = today + timedelta(days=m["day_offset"])
    at = lambda hhmm: datetime.combine(day, time.fromisoformat(hhmm), tz)  # noqa: E731
    return m | {"kam_id": m["kam_id"].lower(), "starts_at": at(m["start"]),
                "ends_at": at(m["end"])}


class PgKamTools(_Directory):
    """Calendar and competitor updates from Postgres; KAMs are support_reviewer rows."""

    def __init__(self, store: Any, merchants: dict[str, dict[str, Any]], now: datetime,
                 tz: ZoneInfo) -> None:
        super().__init__(merchants, now)
        self.store, self.tz = store, tz

    def migrate(self) -> None:
        with self.store.pool.connection() as conn:
            conn.execute(files("teammates.kam").joinpath("schema.sql").read_text("utf-8"))

    def seed_demo(self, meetings: list[dict[str, Any]], feed: list[dict[str, Any]],
                  sources: list[Any], now: datetime) -> None:
        """Demo meetings for today and tomorrow, and the mocked feed fetch, idempotent.
        ponytail: the feed is a fixed list; a real fetcher polls the [[sources]] URLs."""
        today = now.astimezone(self.tz).date()
        with self.store.pool.connection() as conn:
            cur = conn.cursor()
            cur.executemany(
                "INSERT INTO kam_meeting (kam_email, starts_at, ends_at, merchant_id, title,"
                " location) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [(r["kam_id"], r["starts_at"], r["ends_at"], r["merchant_id"], r["title"],
                  r["location"]) for r in (_meeting_row(m, today, self.tz) for m in meetings)])
            cur.executemany(
                "INSERT INTO competitor_update (source, competitor, title, body, url,"
                " published_at) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (url) DO NOTHING",
                [(i["source"], i["competitor"], i["title"], i["body"], i["url"],
                  i["published_at"]) for i in fixed_source_items(feed, sources, now)])

    def kams(self) -> set[str]:
        return {r["email"] for r in self.store.by_role("kam")}

    def get_meetings(self, kam_id: str, day: date) -> list[dict[str, Any]]:
        self._known(kam_id)
        lo = datetime.combine(day, time(), self.tz)
        with self.store.pool.connection() as conn:
            return conn.execute(
                "SELECT starts_at, ends_at, title, merchant_id, location FROM kam_meeting"
                " WHERE kam_email = %s AND starts_at >= %s AND starts_at < %s"
                " ORDER BY starts_at", (kam_id, lo, lo + timedelta(days=1))).fetchall()

    def get_competitor_updates(self, since: datetime) -> list[dict[str, Any]]:
        with self.store.pool.connection() as conn:
            return conn.execute(
                "SELECT competitor, source, title, body, url, published_at"
                " FROM competitor_update WHERE published_at >= %s"
                " ORDER BY published_at DESC", (since,)).fetchall()
