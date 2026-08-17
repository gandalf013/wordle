"""Tests for skill.py (Wordlebot-style skill/luck scores)."""

import math

import pytest

from analysis import GuessAnalysis
from skill import luck, luck_score, skill, skill_score


def _analysis(guess, *, entropy=0.0, buckets=None, bucket_counts=None):
    return GuessAnalysis(
        guess=guess,
        entropy=entropy,
        worst_case_size=0,
        expected_size=0.0,
        is_possible_solution=False,
        buckets=buckets,
        bucket_counts=bucket_counts,
    )


class TestSkillScore:
    def test_top_guess_scores_100(self):
        assert skill_score(3.0, [1.0, 2.0, 3.0]) == pytest.approx(100.0)

    def test_bottom_guess_scores_frac(self):
        assert skill_score(1.0, [1.0, 2.0, 3.0]) == pytest.approx(100.0 / 3.0)

    def test_middle_guess(self):
        assert skill_score(2.0, [1.0, 2.0, 3.0]) == pytest.approx(200.0 / 3.0)

    def test_ties_count_inclusively(self):
        assert skill_score(2.0, [2.0, 2.0, 2.0]) == pytest.approx(100.0)

    def test_empty(self):
        assert skill_score(2.0, []) == 0.0


class TestLuckScore:
    def test_largest_bucket_scores_zero(self):
        # buckets [5, 3, 2] over N=10; landing in the largest (5) is worst.
        assert luck_score([5, 3, 2], 5) == pytest.approx(0.0)

    def test_smallest_bucket_is_lucky(self):
        assert luck_score([5, 3, 2], 2) > luck_score([5, 3, 2], 3) > luck_score([5, 3, 2], 5)

    def test_normalized_value(self):
        # info_min = log2(10) - log2(5) = 1; info_max = log2(10); k=2 gives log2(5).
        info_max = math.log2(10)
        info_min = info_max - math.log2(5)
        info_actual = info_max - math.log2(2)
        expected = 100.0 * (info_actual - info_min) / (info_max - info_min)
        assert luck_score([5, 3, 2], 2) == pytest.approx(expected)

    def test_single_survivor_scores_100(self):
        # A perfect singleton split: every outcome is the answer.
        assert luck_score([1, 1, 1, 1], 1) == pytest.approx(100.0)

    def test_single_bucket_scores_zero(self):
        assert luck_score([4], 4) == pytest.approx(0.0)

    def test_empty(self):
        assert luck_score([], 0) == 0.0

    def test_unknown_bucket_size_raises(self):
        with pytest.raises(ValueError):
            luck_score([5, 3, 2], 4)


class TestWrappers:
    def test_skill_wrapper(self):
        guess = _analysis("slate", entropy=2.0)
        others = [_analysis("crane", entropy=1.0), _analysis("tarse", entropy=3.0)]
        assert skill(guess, [guess, *others]) == pytest.approx(200.0 / 3.0)

    def test_luck_wrapper_uses_buckets(self):
        guess = _analysis("slate", buckets={0: ["a", "b", "c"], 1: ["d", "e"]})
        assert luck(guess, 0) == pytest.approx(luck_score([3, 2], 3))
        assert luck(guess, 1) == pytest.approx(luck_score([3, 2], 2))

    def test_luck_wrapper_falls_back_to_bucket_counts(self):
        guess = _analysis("slate", bucket_counts=((0, 3), (1, 2)))
        assert luck(guess, 1) == pytest.approx(luck_score([3, 2], 2))

    def test_luck_wrapper_unknown_score_raises(self):
        guess = _analysis("slate", buckets={0: ["a", "b"]})
        with pytest.raises(ValueError):
            luck(guess, 5)

    def test_luck_wrapper_no_bucket_data_raises(self):
        guess = _analysis("slate")
        with pytest.raises(ValueError):
            luck(guess, 0)
