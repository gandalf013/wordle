"""Tests for engine.SolverEngine: the I/O-free game state machine that
replaced Game. Ported from the old Game-based TestPlayOneRound/
TestGetGuessScore tests, minus everything that was about input()/print()
(that behavior now lives in cli.py, tested separately in test_cli.py).
"""

import pytest

import scoring
from engine import RoundOutcome, SolverEngine
from strategies import EntropyStrategy


class TestSuggest:
    def test_uses_strategy_to_pick_best_guess(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        assert engine.suggest().guess == "aa"

    def test_initial_guess_overrides_the_strategy_on_the_first_round(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="bb"
        )
        assert engine.suggest().guess == "bb"

    def test_initial_guess_does_not_override_after_the_first_round(self):
        engine = SolverEngine(
            ["aa", "ab", "ba", "bb"],
            ["aa", "ab", "ba", "bb"],
            EntropyStrategy(),
            initial_guess="aa",
        )
        engine.apply_score("aa", scoring.get_score("aa", "ba"))
        # Once history is non-empty, initial_guess no longer applies --
        # suggest() falls back to the strategy over the narrowed pool.
        assert engine.suggest().guess != "aa" or "aa" not in engine.candidates

    def test_caches_the_first_round_suggestion(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        first = engine.suggest()
        second = engine.suggest()
        assert first is second

    def test_reset_preserves_the_cached_first_round_suggestion(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        first = engine.suggest()
        engine.apply_score("aa", scoring.get_score("aa", "ab"))
        engine.reset()
        assert engine.suggest() is first


class TestAnalysisCache:
    def test_get_analyses_caches_and_invalidates_on_apply_score(self):
        engine = SolverEngine(["aa", "ab", "ba"], ["aa", "ab", "ba"], EntropyStrategy())
        first_analyses = engine.get_analyses()
        second_analyses = engine.get_analyses()
        assert first_analyses is second_analyses

        engine.apply_score("aa", scoring.get_score("aa", "ab"))
        third_analyses = engine.get_analyses()
        assert third_analyses is not first_analyses

    def test_reset_invalidates_analyses_cache(self):
        engine = SolverEngine(["aa", "ab", "ba"], ["aa", "ab", "ba"], EntropyStrategy())
        first_analyses = engine.get_analyses()
        engine.apply_score("aa", scoring.get_score("aa", "ab"))
        engine.reset()
        second_analyses = engine.get_analyses()
        assert second_analyses is not first_analyses


class TestAnalyze:
    def test_does_not_touch_candidates_or_history(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        result = engine.analyze("zz")
        assert result.guess == "zz"
        assert engine.candidates == ["aa", "ab", "ac"]
        assert engine.history == []

    def test_reuses_cached_analyses_when_buckets_not_needed(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        cached = engine.get_analyses()
        result = engine.analyze("aa", include_buckets=False)
        assert result is cached[1]  # same object -- no re-scoring

    def test_analyze_without_buckets_returns_the_cached_object(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        engine.get_analyses()
        result = engine.analyze("aa", include_buckets=False)
        assert result.buckets is None  # cached analyses don't carry buckets

    def test_analyze_with_buckets_recomputes(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        engine.get_analyses()
        result = engine.analyze("aa")
        assert result.guess == "aa"
        assert result.buckets is not None


class TestApplyScore:
    def test_narrows_candidates_to_the_matching_score(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        result = engine.apply_score("aa", scoring.get_score("aa", "ab"))
        assert result.outcome == RoundOutcome.CONTINUE
        assert engine.candidates == [
            w for w in ["aa", "ab", "ac"] if scoring.get_score("aa", w) == scoring.get_score("aa", "ab")
        ]

    def test_records_history(self):
        engine = SolverEngine(["ax", "aa"], ["aa", "ab", "ac"], EntropyStrategy())
        score = scoring.get_score("aa", "ab")
        engine.apply_score("aa", score)
        assert engine.history == [("aa", score)]

    def test_deduces_a_unique_remaining_candidate_as_solved(self):
        engine = SolverEngine(
            ["aa", "ab", "ba", "bb"], ["aa", "ab", "ba", "bb"], EntropyStrategy()
        )
        result = engine.apply_score("aa", scoring.get_score("aa", "bb"))
        assert result.outcome == RoundOutcome.SOLVED
        assert result.solution == "bb"
        # history records only the move actually played; the implied final
        # "guess it" move is accounted for by guesses_used instead.
        assert engine.history == [("aa", scoring.get_score("aa", "bb"))]
        assert result.guesses_used == 2
        assert engine.candidates == ["bb"]

    def test_guessing_the_answer_directly_does_not_duplicate_history(self):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        perfect_score = scoring.get_score_num([scoring.Score.GREEN] * 2)
        result = engine.apply_score("bb", perfect_score)
        assert result.outcome == RoundOutcome.SOLVED
        assert result.solution == "bb"
        assert result.guesses_used == 1
        assert engine.history == [("bb", perfect_score)]

    def test_no_candidates_match_score_is_an_error(self):
        # guess "aa" can only ever score 8 (vs "aa") or 0 (vs "bb"); "01"
        # (1) is a well-formed but unachievable score for this pool.
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())
        result = engine.apply_score("aa", 1)
        assert result.outcome == RoundOutcome.ERROR
        assert result.candidates_remaining == 0
        # The pool is left untouched on error, not cleared.
        assert engine.candidates == ["aa", "bb"]

    def test_error_still_records_history(self):
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())
        engine.apply_score("aa", 1)
        assert engine.history == [("aa", 1)]


class TestReset:
    def test_restores_candidates_and_clears_history(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="bb"
        )
        engine.apply_score("bb", scoring.get_score("bb", "aa"))
        assert engine.candidates != ["aa", "bb"]

        engine.reset()
        assert engine.candidates == ["aa", "bb"]
        assert engine.history == []
