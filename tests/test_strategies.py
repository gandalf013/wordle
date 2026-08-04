"""Tests for strategies.py.

EntropyStrategy(weighted=False)'s expected winners below (including the
real-word-list golden value "tarse") were originally cross-checked live
against the old Game.find_best_guess before Game was retired in favor of
engine.SolverEngine -- see the "EntropyStrategy(weighted=False) matches
Game.find_best_guess" note in REFACTOR_PLAN.md's migration log for that
history. They're asserted directly here now rather than re-deriving them
from a live comparison each run. ExpectedPoolSizeStrategy and MinimaxStrategy
have no Game equivalent to cross-check against (they're new), so their
tests verify the ranking directly against hand-computed expectations.
"""

from pathlib import Path

import pytest

import analysis
from analysis import GuessAnalysis
from strategies import EntropyStrategy, ExpectedPoolSizeStrategy, MinimaxStrategy
from wordlists import parse_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def _analysis(
    guess,
    *,
    entropy=0.0,
    worst_case_size=0,
    expected_size=0.0,
    is_possible_solution=False,
    weighted_entropy=None,
    weighted_expected_size=None,
    solution_probability=None,
):
    """Build a GuessAnalysis directly rather than deriving it from real word
    scoring, for tests that need specific (e.g. exactly-tied) field values
    that are otherwise impractical to hit reliably via real buckets."""
    return GuessAnalysis(
        guess=guess,
        buckets={},
        entropy=entropy,
        worst_case_size=worst_case_size,
        expected_size=expected_size,
        is_possible_solution=is_possible_solution,
        weighted_entropy=weighted_entropy,
        weighted_expected_size=weighted_expected_size,
        solution_probability=solution_probability,
    )


class TestEntropyStrategyMatchesGame:
    def test_picks_the_higher_entropy_guess(self):
        guesses, targets = ["ax", "aa"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets)
        ranked = EntropyStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_tie_break_prefers_a_possible_solution(self):
        # "aa" and "bb" have identical entropy against these targets; only
        # "aa" is an actual candidate solution.
        guesses, targets = ["bb", "aa"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets)
        ranked = EntropyStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_rank_returns_all_analyses_best_first(self):
        guesses, targets = ["ax", "aa", "bb"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets)
        ranked = EntropyStrategy().rank(results)
        assert {r.guess for r in ranked} == {r.guess for r in results}
        assert len(ranked) == len(results)
        assert ranked[0].guess == "aa"


@pytest.mark.slow
class TestEntropyStrategyRealWordList:
    def test_round_one_matches_game_golden_value(self):
        with open(REPO_ROOT / "words.wordle.txt") as fp:
            wl = parse_file(fp)
        guesses = sorted(set(wl.target) | set(wl.extra))
        targets = wl.target

        results = analysis.analyze_all(guesses, targets, use_cache=True)
        ranked = EntropyStrategy().rank(results)
        assert ranked[0].guess == "tarse"
        assert ranked[0].entropy == pytest.approx(5.948974509955522)


class TestEntropyStrategyWeighted:
    # Same hand-built pool as TestWeightedAnalyze in test_analysis.py: "dc"
    # finely splits low-weight chaff (higher raw entropy) but lumps the two
    # high-weight plausible answers together; "bb" does the reverse.
    TARGETS = ["aa", "ab", "ca", "cb", "cc", "cd"]
    WEIGHTS = {
        "aa": 100.0,
        "ab": 100.0,
        "ca": 0.001,
        "cb": 0.001,
        "cc": 0.001,
        "cd": 0.001,
    }

    def test_weighted_ranking_prefers_the_guess_that_separates_likely_answers(self):
        results = analysis.analyze_all(
            ["dc", "bb"], self.TARGETS, weights=self.WEIGHTS
        )
        assert EntropyStrategy(weighted=False).rank(results)[0].guess == "dc"
        assert EntropyStrategy(weighted=True).rank(results)[0].guess == "bb"

    def test_weighted_tie_break_prefers_higher_solution_probability(self):
        # Hand-built rather than derived from real word scoring: three
        # analyses tied on weighted_entropy, only two of which are
        # candidates, with candidate_high's solution_probability higher
        # than candidate_low's. Constructing GuessAnalysis directly (rather
        # than searching for real words that happen to tie exactly) is what
        # makes this deterministic -- exact float ties are otherwise hard
        # to hit reliably via real bucket scoring.
        non_candidate = _analysis(
            "xy", entropy=1.0, weighted_entropy=1.0, is_possible_solution=False
        )
        candidate_low = _analysis(
            "lo",
            entropy=1.0,
            weighted_entropy=1.0,
            is_possible_solution=True,
            solution_probability=0.2,
        )
        candidate_high = _analysis(
            "hi",
            entropy=1.0,
            weighted_entropy=1.0,
            is_possible_solution=True,
            solution_probability=0.8,
        )

        # Weighted: scans the whole tied block and picks the highest
        # solution_probability among candidates, not the first one found.
        ranked = EntropyStrategy(weighted=True).rank(
            [non_candidate, candidate_low, candidate_high]
        )
        assert ranked[0].guess == "hi"

        # Unweighted: stops at the first candidate found in the tied block
        # (list order preserved for ties), matching find_best_guess.
        ranked = EntropyStrategy(weighted=False).rank(
            [non_candidate, candidate_low, candidate_high]
        )
        assert ranked[0].guess == "lo"


class TestExpectedPoolSizeStrategy:
    def test_minimizes_expected_remaining_pool_size(self):
        # "aa" splits ["aa","ab","ac"] into a 1-word and a 2-word bucket
        # (E = (1*1 + 2*2)/3); "ax" reveals nothing, so its one bucket
        # covers all 3 (E = 3). "aa" should win.
        results = analysis.analyze_all(["ax", "aa"], ["aa", "ab", "ac"])
        ranked = ExpectedPoolSizeStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_weighted_mode_can_pick_a_different_winner_than_uniform(self):
        # Hand-built: "small_raw" has the smaller raw expected_size but a
        # larger weighted_expected_size than "small_weighted", and vice
        # versa -- exactly the disagreement weighted=True exists to catch.
        small_raw = _analysis("sr", expected_size=1.0, weighted_expected_size=9.0)
        small_weighted = _analysis("sw", expected_size=2.0, weighted_expected_size=1.0)

        assert (
            ExpectedPoolSizeStrategy(weighted=False).rank([small_raw, small_weighted])[
                0
            ].guess
            == "sr"
        )
        assert (
            ExpectedPoolSizeStrategy(weighted=True).rank([small_raw, small_weighted])[
                0
            ].guess
            == "sw"
        )


class TestMinimaxStrategy:
    def test_minimizes_worst_case_bucket_size(self):
        # "ax" reveals nothing -> one bucket of size 3 (worst case 3).
        # "aa" splits into buckets of size 1 and 2 (worst case 2).
        results = analysis.analyze_all(["ax", "aa"], ["aa", "ab", "ac"])
        ranked = MinimaxStrategy().rank(results)
        assert ranked[0].guess == "aa"
        assert ranked[0].worst_case_size == 2
