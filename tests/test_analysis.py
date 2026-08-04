"""Tests for analysis.py.

The core correctness claim of this module is "matches Game.get_all_censuses
and Game.get_all_entropy", since it's meant as a direct (eventually
weight-aware) replacement for them -- so most tests here cross-check against
Game rather than re-deriving expected values by hand.
"""

import math
from pathlib import Path

import pytest

import analysis
import fast_scoring
from wordle import Game
from wordlists import parse_file

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_score_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fast_scoring, "CACHE_DIR", tmp_path / "wordle_cache")


class TestAnalyze:
    def test_buckets_group_words_by_score(self):
        result = analysis.analyze("aa", ["aa", "ab", "ac", "ba"])
        g = Game(["aa"], ["aa"])
        assert result.buckets == {
            g.get_score("aa", "aa"): ["aa"],
            g.get_score("aa", "ab"): ["ab", "ac"],
            g.get_score("aa", "ba"): ["ba"],
        }

    def test_entropy_matches_game_for_perfectly_split_pool(self):
        result = analysis.analyze("aa", ["aa", "ab", "ac"])
        expected = -(1 / 3 * math.log2(1 / 3) + 2 / 3 * math.log2(2 / 3))
        assert result.entropy == pytest.approx(expected)

    def test_zero_entropy_when_guess_reveals_nothing(self):
        result = analysis.analyze("ax", ["aa", "ab", "ac"])
        assert result.entropy == pytest.approx(0.0)

    def test_worst_case_size_is_largest_bucket(self):
        result = analysis.analyze("ax", ["aa", "ab", "ac"])
        assert result.worst_case_size == 3

    def test_expected_size_is_size_weighted_average(self):
        # "aa" vs ["aa", "ab", "ac"]: buckets of size 1 and 2 -> E = (1*1 + 2*2)/3
        result = analysis.analyze("aa", ["aa", "ab", "ac"])
        assert result.expected_size == pytest.approx((1 * 1 + 2 * 2) / 3)

    def test_is_possible_solution_true_when_guess_in_pool(self):
        result = analysis.analyze("aa", ["aa", "ab", "ac"])
        assert result.is_possible_solution is True

    def test_is_possible_solution_false_when_guess_not_in_pool(self):
        result = analysis.analyze("zz", ["aa", "ab", "ac"])
        assert result.is_possible_solution is False

    def test_weighted_fields_default_to_none(self):
        result = analysis.analyze("aa", ["aa", "ab", "ac"])
        assert result.weighted_entropy is None
        assert result.weighted_expected_size is None
        assert result.solution_probability is None


class TestAnalyzeAll:
    def test_entropy_matches_game_get_all_censuses(self):
        guesses = ["ax", "aa"]
        targets = ["aa", "ab", "ac"]
        g = Game(guesses, targets)
        censuses = g.get_all_censuses()
        game_entropy = g.get_all_entropy(censuses)

        results = analysis.analyze_all(guesses, targets)
        for result, expected_entropy in zip(results, game_entropy):
            assert result.entropy == pytest.approx(expected_entropy)

    def test_buckets_match_across_all_guesses(self):
        guesses = ["ax", "aa", "bb"]
        targets = ["aa", "ab", "ac", "ba", "bb"]
        g = Game(guesses, targets)

        results = analysis.analyze_all(guesses, targets)
        for guess, result in zip(guesses, results):
            expected_buckets: dict[int, list[str]] = {}
            for target in targets:
                score = g.get_score(guess, target)
                expected_buckets.setdefault(score, []).append(target)
            assert result.buckets == expected_buckets

    def test_returns_one_analysis_per_guess_in_order(self):
        guesses = ["ax", "aa", "bb"]
        results = analysis.analyze_all(guesses, ["aa", "ab", "ba", "bb"])
        assert [r.guess for r in results] == guesses

    def test_use_cache_populates_the_score_matrix_cache(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(fast_scoring, "CACHE_DIR", cache_dir)
        analysis.analyze_all(["aa", "ab"], ["aa", "ab", "ba", "bb"], use_cache=True)
        assert len(list(cache_dir.glob("*.npy"))) == 1

    def test_no_cache_by_default(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(fast_scoring, "CACHE_DIR", cache_dir)
        analysis.analyze_all(["aa", "ab"], ["aa", "ab", "ba", "bb"])
        assert list(cache_dir.glob("*.npy")) == []


@pytest.mark.slow
class TestRealWordList:
    @pytest.fixture(autouse=True)
    def isolated_score_cache(self):
        # Overrides the module-level isolation fixture: this test's job is
        # to prove parity against Game on the real, full-size word list.
        yield

    def test_round_one_entropy_matches_game_for_every_guess(self):
        with open(REPO_ROOT / "words.wordle.txt") as fp:
            wl = parse_file(fp)
        guesses = sorted(set(wl.target) | set(wl.extra))
        targets = wl.target

        g = Game(guesses, targets)
        censuses = g.get_all_censuses()
        game_entropy = g.get_all_entropy(censuses)

        results = analysis.analyze_all(guesses, targets, use_cache=True)
        for result, expected_entropy in zip(results, game_entropy):
            assert result.entropy == pytest.approx(expected_entropy)

        best = max(results, key=lambda r: r.entropy)
        assert best.guess == "tarse"
        assert best.entropy == pytest.approx(5.948974509955522)
