"""VOI limits used by scripts/check.py."""
from gpu_waterz.limits import AFF_THRESHOLDS, SLACK, grade_voi, voi_limits


def test_four_thresholds():
    assert AFF_THRESHOLDS == (0.2, 0.3, 0.4, 0.5)


def test_slack_on_stock_bases():
    lim = voi_limits()
    ok, sl, ml = grade_voi(0.4538, 0.2411, 0.3)
    assert ok
    assert sl == 0.4538 + SLACK
    assert ml == 0.2411 + SLACK
    assert not grade_voi(sl + 1e-6, 0.2411, 0.3)[0]
    assert set(lim) == set(AFF_THRESHOLDS)
