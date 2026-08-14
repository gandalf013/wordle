"""Tests for cli.py: the REPL grammar (parse_command) and the interactive
I/O loop built on SolverEngine (play_one_round/run_interactive), plus the
--strategy/--weighted wiring in build_strategy.

test_already_solved_round_is_a_noop_quit from the pre-refactor Game-based
suite isn't ported: it exercised Game's own defensive "already solved,
don't play again" guard (self.found_solution), which was an implementation
detail of Game's persistent state, not part of the engine/cli split's
contract -- run_interactive's loop structure never calls play_one_round
again after a SOLVED/ERROR outcome without going through reset() first, so
the scenario that guard protected against can't arise here.
"""

from unittest.mock import patch

import pytest

import scoring
from cli import (
    Analyze,
    Back,
    Buckets,
    Help,
    LoopState,
    OverrideGuess,
    Pool,
    Quit,
    Restart,
    ShowScore,
    Top,
    build_strategy,
    main,
    parse_command,
    play_one_round,
    run_interactive,
)
from engine import SolverEngine
from strategies import (
    EntropyStrategy,
    ExpectedPoolSizeStrategy,
    MaxBinsBalanceStrategy,
    MinimaxStrategy,
    NumBinsStrategy,
)


def _score_digits(score: int, n: int) -> str:
    return "".join(str(int(x)) for x in scoring.get_score_list(score, n))


@pytest.fixture
def engine():
    return SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())


class TestParseCommand:
    def test_score_digits(self):
        assert parse_command("01", 2) == ShowScore(1)

    def test_wrong_length_score_is_none(self):
        assert parse_command("0", 2) is None

    def test_non_ternary_digits_are_none(self):
        assert parse_command("03", 2) is None

    def test_override_guess(self):
        assert parse_command("!bb", 2) == OverrideGuess("bb")

    def test_override_guess_wrong_length_is_none(self):
        assert parse_command("!b", 2) is None

    def test_override_guess_is_lowercased(self):
        assert parse_command("!BB", 2) == OverrideGuess("bb")

    def test_analyze_question_mark_form(self):
        assert parse_command("?bb", 2) == Analyze("bb")

    def test_analyze_word_form(self):
        assert parse_command("analyze bb", 2) == Analyze("bb")

    def test_analyze_missing_word_is_none(self):
        assert parse_command("analyze", 2) is None

    def test_analyze_wrong_length_word_is_none(self):
        assert parse_command("analyze b", 2) is None

    def test_buckets_no_word_uses_none(self):
        assert parse_command("buckets", 2) == Buckets(None)

    def test_buckets_with_word(self):
        assert parse_command("buckets bb", 2) == Buckets("bb")

    def test_buckets_wrong_length_word_is_none(self):
        assert parse_command("buckets b", 2) is None

    def test_top_default_n(self):
        assert parse_command("top", 2) == Top(10)

    def test_top_with_explicit_n(self):
        assert parse_command("top 5", 2) == Top(5)

    def test_top_invalid_n_is_none(self):
        assert parse_command("top abc", 2) is None

    def test_pool_default_shows_all(self):
        assert parse_command("pool", 2) == Pool(None)

    def test_pool_with_explicit_n(self):
        assert parse_command("pool 5", 2) == Pool(5)

    def test_pool_invalid_n_is_none(self):
        assert parse_command("pool abc", 2) is None

    def test_pool_non_positive_n_is_none(self):
        assert parse_command("pool 0", 2) is None
        assert parse_command("pool -1", 2) is None

    def test_back_default_n(self):
        assert parse_command("b", 2) == Back(1)
        assert parse_command("back", 2) == Back(1)

    def test_back_with_explicit_n(self):
        assert parse_command("b 2", 2) == Back(2)
        assert parse_command("back 3", 2) == Back(3)

    def test_back_invalid_n_is_none(self):
        assert parse_command("b 0", 2) is None
        assert parse_command("back -1", 2) is None
        assert parse_command("back abc", 2) is None

    def test_restart_short_and_long_form(self):
        assert parse_command("r", 2) == Restart()
        assert parse_command("restart", 2) == Restart()

    def test_quit_short_and_long_form(self):
        assert parse_command("q", 2) == Quit()
        assert parse_command("quit", 2) == Quit()

    def test_commands_are_case_insensitive(self):
        assert parse_command("RESTART", 2) == Restart()
        assert parse_command("QUIT", 2) == Quit()

    def test_empty_input_is_none(self):
        assert parse_command("", 2) is None
        assert parse_command("   ", 2) is None

    def test_unrecognized_input_is_none(self):
        assert parse_command("xyz", 2) is None

    def test_bare_question_mark_is_help(self):
        assert parse_command("?", 2) == Help()


class TestPlayOneRoundAutomatic:
    def test_automatic_solve_deduces_unique_remaining_candidate(self):
        engine = SolverEngine(
            ["aa", "ab", "ba", "bb"],
            ["aa", "ab", "ba", "bb"],
            EntropyStrategy(),
            initial_guess="aa",
        )
        state = play_one_round(engine, automatic=True, solution="bb")
        assert state == LoopState.SOLVED
        # history records only the real guess; the implied final all-green
        # move is rendered by _commit, not stored in engine state.
        assert engine.history == [("aa", scoring.get_score("aa", "bb"))]

    def test_automatic_solve_renders_the_implied_final_guess(self, capsys):
        engine = SolverEngine(
            ["aa", "ab", "ba", "bb"],
            ["aa", "ab", "ba", "bb"],
            EntropyStrategy(),
            initial_guess="aa",
        )
        play_one_round(engine, automatic=True, solution="bb")
        out = capsys.readouterr().out
        assert "bb 🟩🟩" in out  # the deduced answer shown as a green row

    def test_guessing_the_answer_directly_does_not_duplicate_history_entry(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="bb"
        )
        state = play_one_round(engine, automatic=True, solution="bb")
        assert state == LoopState.SOLVED
        assert engine.history == [("bb", scoring.get_score_num([scoring.Score.GREEN] * 2))]


class TestPlayOneRoundInteractive:
    def test_no_candidates_match_score_is_an_error(self):
        # guess "aa" can only ever score 8 (vs "aa") or 0 (vs "bb"); "01"
        # (1) is a well-formed but unachievable score for this pool.
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())
        with patch("builtins.input", return_value="01"):
            state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.ERROR

    def test_interactive_eof_quits(self):
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy(), initial_guess="aa")
        with patch("builtins.input", side_effect=EOFError):
            assert play_one_round(engine, automatic=False, solution=None) == LoopState.QUIT

    def test_unparseable_input_reprompts_instead_of_crashing(self):
        engine = SolverEngine(["aa"], ["aa", "bb"], EntropyStrategy())
        with patch("builtins.input", side_effect=["xyz", "01"]):
            state = play_one_round(engine, automatic=False, solution=None)
        # "xyz" is silently ignored (reprompted); "01" (score 1) is then
        # read as a real ShowScore, which "aa" can't achieve against this pool.
        assert state == LoopState.ERROR

    def test_override_guess_then_score_commits_with_the_overridden_guess(self):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        target_score = scoring.get_score("bb", "aa")
        with patch("builtins.input", side_effect=["!bb", _score_digits(target_score, 2)]):
            state = play_one_round(engine, automatic=False, solution=None)
        # Only "aa" matches that score against "bb", so the 2-word pool is
        # immediately solved; history records the one real guess played.
        assert state == LoopState.SOLVED
        assert engine.history == [("bb", target_score)]

    def test_override_guess_then_empty_input_commits_via_solution(self):
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="aa"
        )
        with patch("builtins.input", side_effect=["!bb", ""]):
            state = play_one_round(engine, automatic=False, solution="bb")
        assert state == LoopState.SOLVED
        assert engine.history == [("bb", scoring.get_score_num([scoring.Score.GREEN] * 2))]

    def test_solution_overrides_a_manually_typed_score(self):
        # With a known solution, whatever digits the user types are
        # discarded in favor of the solution-derived score -- matching the
        # priority the old get_guess_score gave a known solution.
        engine = SolverEngine(
            ["aa", "bb"], ["aa", "bb"], EntropyStrategy(), initial_guess="aa"
        )
        with patch("builtins.input", return_value="00"):
            state = play_one_round(engine, automatic=False, solution="bb")
        assert state == LoopState.SOLVED
        # Only "bb" matches "aa"'s score against the solution, so the
        # 2-word pool is immediately solved -- history records the one real
        # guess ("aa"), and the final move is implied by the result.
        assert engine.history == [("aa", scoring.get_score("aa", "bb"))]

    def test_analyze_buckets_and_top_do_not_commit(self, capsys):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        with patch(
            "builtins.input", side_effect=["?bb", "buckets", "top", "restart"]
        ):
            state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.RESTART
        assert engine.history == []
        assert engine.candidates == ["aa", "bb"]
        out = capsys.readouterr().out
        assert "guess" in out  # format_top_guesses header, from ?bb and top

    def test_pool_does_not_commit(self, capsys):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        with patch("builtins.input", side_effect=["pool", "restart"]):
            state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.RESTART
        assert engine.history == []
        assert engine.candidates == ["aa", "bb"]
        out = capsys.readouterr().out
        assert "aa" in out and "bb" in out

    def test_bare_question_mark_prints_help_and_does_not_commit(self, capsys):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        with patch("builtins.input", side_effect=["?", "restart"]):
            state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.RESTART
        assert engine.history == []
        out = capsys.readouterr().out
        assert "Commands:" in out

    def test_back_command_undos_move_and_continues(self, caplog):
        engine = SolverEngine(["aa", "ab", "ba"], ["aa", "ab", "ba"], EntropyStrategy())
        score = scoring.get_score("aa", "ab")
        engine.apply_score("aa", score)
        assert len(engine.candidates) < 3

        with caplog.at_level("INFO"):
            with patch("builtins.input", side_effect=["back", "restart"]):
                state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.RESTART
        assert engine.history == []
        assert len(engine.candidates) == 3
        assert "Backed up 1 move(s)" in caplog.text

    def test_back_on_empty_history_logs_and_reprompts(self, caplog):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        with caplog.at_level("INFO"):
            with patch("builtins.input", side_effect=["back", "restart"]):
                state = play_one_round(engine, automatic=False, solution=None)
        assert state == LoopState.RESTART
        assert "No moves to undo" in caplog.text



class TestRunInteractive:
    def test_restart_resets_the_engine_and_continues(self):
        engine = SolverEngine(["aa", "bb"], ["aa", "bb"], EntropyStrategy())
        # unparseable "" reprompts, "restart" ends the round, then EOF on
        # the next round's prompt ends the loop.
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


class TestBuildStrategy:
    def test_entropy_default_is_unweighted(self):
        strategy = build_strategy("entropy", False)
        assert isinstance(strategy, EntropyStrategy)
        assert strategy.weighted is False

    def test_entropy_weighted(self):
        strategy = build_strategy("entropy", True)
        assert isinstance(strategy, EntropyStrategy)
        assert strategy.weighted is True

    def test_expected_pool_size_weighted(self):
        strategy = build_strategy("expected-pool-size", True)
        assert isinstance(strategy, ExpectedPoolSizeStrategy)
        assert strategy.weighted is True

    def test_minimax_ignores_weighted_flag_with_a_warning(self, caplog):
        with caplog.at_level("WARNING"):
            strategy = build_strategy("minimax", True)
        assert isinstance(strategy, MinimaxStrategy)
        assert "weighted" in caplog.text.lower()

    def test_num_bins_default_is_built(self):
        strategy = build_strategy("num-bins", False)
        assert isinstance(strategy, NumBinsStrategy)

    def test_num_bins_ignores_weighted_flag_with_a_warning(self, caplog):
        with caplog.at_level("WARNING"):
            strategy = build_strategy("num-bins", True)
        assert isinstance(strategy, NumBinsStrategy)
        assert "weighted" in caplog.text.lower()

    def test_max_bins_balance_default_is_unweighted(self):
        strategy = build_strategy("max-bins-balance", False)
        assert isinstance(strategy, MaxBinsBalanceStrategy)
        assert strategy.weighted is False

    def test_max_bins_balance_weighted(self):
        strategy = build_strategy("max-bins-balance", True)
        assert isinstance(strategy, MaxBinsBalanceStrategy)
        assert strategy.weighted is True


class TestMainIntegration:
    def test_weighted_and_strategy_flags_run_end_to_end(self, tmp_path):
        wordfile = tmp_path / "words.txt"
        wordfile.write_text("aa 5\nab 1\nba 1\nbb 1\n")

        with patch("builtins.input", side_effect=EOFError):
            main(
                [
                    str(wordfile),
                    "-a",
                    "-s",
                    "bb",
                    "--strategy",
                    "expected-pool-size",
                    "--weighted",
                ]
            )

    def test_num_bins_strategy_runs_end_to_end(self, tmp_path):
        wordfile = tmp_path / "words.txt"
        wordfile.write_text("aa 5\nab 1\nba 1\nbb 1\n")

        with patch("builtins.input", side_effect=EOFError):
            main(
                [
                    str(wordfile),
                    "-a",
                    "-s",
                    "bb",
                    "--strategy",
                    "num-bins",
                ]
            )

    def test_max_bins_balance_strategy_runs_end_to_end(self, tmp_path):
        wordfile = tmp_path / "words.txt"
        wordfile.write_text("aa 5\nab 1\nba 1\nbb 1\n")

        with patch("builtins.input", side_effect=EOFError):
            main(
                [
                    str(wordfile),
                    "-a",
                    "-s",
                    "bb",
                    "--strategy",
                    "max-bins-balance",
                    "--weighted",
                ]
            )
