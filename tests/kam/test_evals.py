from teammates.kam import evals


def test_scorecard_passes_every_threshold():
    report = evals.run()
    failed = {k: m["failures"] for k, m in report["metrics"].items() if not m["passed"]}
    assert report["passed"], failed
    assert set(report["metrics"]) == set(evals.THRESHOLDS)
    info = report["info"]
    assert info["cost_inr_per_digest"] > 0 and 0 < info["cache_hit_rate"] < 1
