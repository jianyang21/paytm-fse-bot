import copy
import math

from teammates.support import risk, threshold

CFG = risk.config()


def row(score, risky, faq=False, overridden=False):
    return {"score": score, "risky": risky, "faq": faq, "overridden": overridden}


def test_p_star_is_review_cost_over_total_cost():
    assert threshold.p_star(CFG) == 0.1  # 1 / (1 + 9)
    cfg = copy.deepcopy(CFG)
    cfg["threshold"] |= {"c_review": 3.0, "c_miss": 1.0}
    assert threshold.p_star(cfg) == 0.75


def test_calibrate_skips_faq_and_override_tickets():
    rows = [row(0, False, faq=True), row(40, True, overridden=True), row(2, False),
            row(7, True), row(8, False), row(23, True)]
    assert threshold.calibrate(rows, 5) == [
        {"lo": 0, "hi": 5, "n": 1, "risky": 0, "p": 0.0},
        {"lo": 5, "hi": 10, "n": 2, "risky": 1, "p": 0.5},
        {"lo": 20, "hi": 25, "n": 1, "risky": 1, "p": 1.0}]


def test_choose_lowest_bin_reaching_p_star():
    bins = [{"lo": 0, "p": 0.05}, {"lo": 5, "p": 0.2}, {"lo": 10, "p": 0.0}, {"lo": 20, "p": 1}]
    assert threshold.choose(bins, 0.1) == 5
    assert threshold.choose(bins, 0.5) == 20
    assert threshold.choose(bins, 1.01) == math.inf  # nothing escalated by score alone


def test_sweep_recall_vs_escalated_share():
    rows = [row(0, False, faq=True), row(3, False), row(8, True), row(30, True, overridden=True)]
    s = {x["t"]: x for x in threshold.sweep(rows, [0, 5, 10])}
    assert s[0] == {"t": 0, "escalated": 0.75, "recall": 1.0}  # FAQ never escalated by score
    assert s[5] == {"t": 5, "escalated": 0.5, "recall": 1.0}
    assert s[10] == {"t": 10, "escalated": 0.25, "recall": 0.5}  # only the override


def test_seed_calibration_gives_a_positive_threshold_that_catches_every_risky_ticket():
    rows = threshold.seed_rows(CFG)
    t = threshold.score_threshold()
    assert 0 < t < 100
    assert threshold.sweep(rows, [t])[0]["recall"] == 1.0
    assert all(r["score"] == 0 for r in rows if r["faq"])


def test_cheaper_misses_raise_the_threshold():
    cfg = copy.deepcopy(CFG)
    cfg["threshold"]["c_miss"] = 1.0  # p* = 0.5
    bins = threshold.calibrate(threshold.seed_rows(cfg), cfg["threshold"]["bin_width"])
    assert threshold.choose(bins, threshold.p_star(cfg)) > threshold.score_threshold()


def test_main_prints_calibration_table_and_sweep(capsys):
    threshold.main()
    out = capsys.readouterr().out
    assert "p* = C_review / (C_review + C_miss)" in out and "P(risky)" in out
    assert "<- Elkan" in out and "Elkan threshold: score >=" in out
