import pytest

from teammates.kam.evals import build


@pytest.fixture
def env():
    """KamDigest over a store the real support desk filled from its seed tickets (fake LLM,
    MockKamTools, fake Redis); env.sent records every delivery, env.fail injects failures."""
    return build()
