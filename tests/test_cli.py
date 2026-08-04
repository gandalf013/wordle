"""Tests for cli.py: the interactive I/O loop built on SolverEngine.
Ported from the old Game-based TestGetGuessScore/TestPlayOneRound tests,
now targeting cli.resolve_score/play_one_round/run_interactive directly
via mocked input() instead of Game methods.

test_already_solved_round_is_a_noop_quit from the old suite isn't ported:
it exercised Game's own defensive "already solved, don't play again" guard
(self.found_solution), which was an implementation detail of Game's
persistent state, not part of the new engine/cli split's contract --
run_interactive's loop structure never calls play_one_round again after a
SOLVED/ERROR outcome without going through reset() first, so the scenario
that guard protected against can't arise here.
"""

import scoring
from cli import LoopState, play_one_round, resolve_score, run_interactive
from engine import SolverEngine
from strategies import EntropyStrategy
from unittest.mock import patch

import pytest


@pytest.fixture
def engine():
    return SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())


class TestResolveScore:
    def test_known_solution_ignores_potential_score(self, engine):
        state, score = resolve_score(engine, "aa", "garbage", solution="bb")
        assert state == LoopState.CONTINUE
        assert score == scoring.get_score("aa", "bb")

    def test_potential_score_wrong_length_is_ignored(self, engine):
        with patch("builtins.input", return_value="01"):
            state, score = resolve_score(engine, "aa", "1", solution=None)
        assert state == LoopState.CONTINUE
        assert score == 1  # int("01", base=3)

    def test_potential_score_valid_is_used_without_prompting(self, engine):
        with patch("builtins.input") as mock_input:
            state, score = resolve_score(engine, "aa", "12", solution=None)
        mock_input.assert_not_called()
        assert state == LoopState.CONTINUE
        assert score == int("12", base=3)

    def test_potential_score_invalid_digits_falls_back_to_prompt(self, engine):
        with patch("builtins.input", return_value="01"):
            state, score = resolve_score(engine, "aa", "1a", solution=None)
        assert state == LoopState.CONTINUE
        assert score == 1

    def test_restart_command(self, engine):
        with patch("builtins.input", return_value="restart"):
            state, score = resolve_score(engine, "aa", None, solution=None)
        assert state == LoopState.RESTART
        assert score is None

    def test_quit_command(self, engine):
        with patch("builtins.input", return_value="quit"):
            state, score = resolve_score(engine, "aa", None, solution=None)
        assert state == LoopState.QUIT
        assert score is None

    def test_eof_quits(self, engine):
        with patch("builtins.input", side_effect=EOFError):
            state, score = resolve_score(engine, "aa", None, solution=None)
        assert state == LoopState.QUIT
        assert score is None


class TestPlayOneRound:
    def test_automatic_solve_deduces_unique_remaining_candidate(self):
        engine = SolverEngine(
            ["aa", "ab", "ba", "bb"],
            ["aa", "ab", "ba", "bb"],
            EntropyStrategy(),
            initial_guess="aa",
        )
        state = play_one_round(engine, automatic=True, solution="bb")
        assert state == LoopState.SOLVED
        # One real guess ("aa") plus a synthesized perfect-score entry for
        # the deduced answer, since "aa" alone didn't score all-green.
        assert engine.history == [
            ("aa", scoring.get_score("aa", "bb")),
            ("bb", scoring.get_score_num([scoring.Score.GREEN] * 2)),
        ]

    def test_guessing_the_answer_directly_does_not_duplicate_history_entry(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="bb"
        )
        state = play_one_round(engine, automatic=True, solution="bb")
        assert state == LoopState.SOLVED
        assert engine.history == [("bb", scoring.get_score_num([scoring.Score.GREEN] * 2))]

    def test_no_candidates_match_score_is_an_error(self):
        # guess "aa" can only ever score 8 (vs "aa") or 0 (vs "bb"); "01"
        # (1) is a well-formed but unachievable score for this pool.
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())
        with patch("builtins.input", return_value="01"):
            state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.ERROR

    def test_interactive_guess_override_replaces_suggestion(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="aa"
        )
        with patch("builtins.input", return_value="bb"):
            state = play_one_round(engine, automatic=False, solution="bb")
        assert state == LoopState.SOLVED
        assert engine.history == [("bb", scoring.get_score_num([scoring.Score.GREEN] * 2))]

    def test_interactive_eof_quits(self):
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy(), initial_guess="aa")
        with patch("builtins.input", side_effect=EOFError):
            assert play_one_round(engine, automatic=False, solution=None) == LoopState.QUIT


class TestRunInteractive:
    def test_restart_resets_the_engine_and_continues(self):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        # "" at the guess prompt (no override), "restart" at the score
        # prompt, then EOF on the next round's guess prompt to end the loop.
        with patch("builtins.input", side_effect=["", "restart", EOFError]):
            run_interactive(engine, automatic=False, solution=None)
        assert engine.candidates == ["aa", "bb"]
        assert engine.history == []

    def test_new_round_prompt_can_continue_or_quit(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="bb"
        )
        # automatic + a fixed solution means the round solves with no
        # per-round input; the only prompts are the two "New round?"s.
        with patch("builtins.input", side_effect=["y", ""]):
            run_interactive(engine, automatic=True, solution="bb")
        assert engine.history == [("bb", scoring.get_score_num([scoring.Score.GREEN] * 2))]
