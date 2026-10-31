from teammates.fse import evals
from tests.fse.conftest import DSN


def test_eval_suite_passes_every_threshold():
    """The full simulated day + fault injection (small size to keep the suite fast)."""
    import pytest

    if not DSN:
        pytest.skip("TEST_DATABASE_URL not set")
    report = evals.run(DSN, n_fses=8, seed_=11)
    failed = {k: m["failures"] for k, m in report["metrics"].items() if not m["passed"]}
    assert report["passed"], failed
