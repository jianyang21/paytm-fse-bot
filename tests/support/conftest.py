from types import SimpleNamespace

import pytest

from teammates.llm.client import TokenUsage
from teammates.llm.fake import FakeLLM
from teammates.support.evals import build, fake_reply
from teammates.support.tools import seed

SEED = {t["id"]: t for t in seed()["tickets"] + seed()["attacks"]}


@pytest.fixture
def env():
    """A Desk on MockTools + MemoryStore + fake Redis + FakeLLM; `sent` records every reply."""
    llm = FakeLLM(fake_reply, TokenUsage(prompt_tokens=400, completion_tokens=80))
    desk, store, tools, sent = build(llm)
    return SimpleNamespace(desk=desk, store=store, tools=tools, sent=sent, llm=llm,
                           kam=store.reviewers["kam@example.com"],
                           agent=store.reviewers["agent@example.com"],
                           run=lambda tid: desk.handle(SEED[tid]["channel"],
                                                       SEED[tid]["payload"]))
