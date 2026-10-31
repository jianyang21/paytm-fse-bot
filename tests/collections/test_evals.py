import pytest

from teammates.collections import evals
from tests.collections.test_collections import DSN


def test_email_eval_passes_every_threshold():
    """The full email eval (small size to keep the suite fast)."""
    if not DSN:
        pytest.skip("TEST_DATABASE_URL not set")
    report = evals.run(DSN, n_borrowers=20, seed_=11)
    failed = {k: m["failures"] for k, m in report["metrics"].items() if not m["passed"]}
    assert report["passed"], failed
