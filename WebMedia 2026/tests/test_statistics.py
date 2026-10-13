import itertools

import numpy as np

from fundrel.liwc import holm
from fundrel.polarity import cliffs_delta


def brute_cliff(x, y):
    gt = sum(a > b for a, b in itertools.product(x, y))
    lt = sum(a < b for a, b in itertools.product(x, y))
    return (gt - lt) / (len(x) * len(y))


def test_cliffs_delta_matches_definition_with_ties():
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 5, 60), rng.integers(1, 6, 45)
    delta, se, (lo, hi) = cliffs_delta(x, y)
    assert abs(delta - brute_cliff(x, y)) < 1e-12
    assert lo < delta < hi and se > 0


def test_paper_delta_is_consistent_with_paper_u():
    # Sec. 4.3: U = 22,820,276 with n = 5,167 (FW.Br) and 8,526 (FT.Br) -> delta = 0.036
    assert round(2 * 22_820_276 / (5167 * 8526) - 1, 3) == 0.036


def test_holm_adjustment():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    expected = np.array([0.03, 0.06, 0.06, 0.02])
    assert np.allclose(holm(p), expected)


def test_cliffs_delta_equals_rescaled_u():
    from scipy import stats
    rng = np.random.default_rng(1)
    x, y = rng.normal(0.1, 1, 300).round(2), rng.normal(0, 1, 250).round(2)
    u = stats.mannwhitneyu(x, y).statistic
    assert abs(cliffs_delta(x, y)[0] - (2 * u / (len(x) * len(y)) - 1)) < 1e-12
