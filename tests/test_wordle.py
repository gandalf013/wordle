"""Behavioral tests for wordle.py, written before any speed refactor.

These pin down current behavior (scoring semantics, encoding, entropy-based
guess selection, the interactive round state machine, and file parsing) so
that a later speed rewrite (vectorized scoring, caching, etc.) can be
validated against a known-good baseline instead of guessing at correctness.

Most tests use tiny synthetic two-letter word lists so they run in
milliseconds and are hand-verifiable. The `slow` tests exercise the real
word lists shipped in the repo and double as a before/after benchmark for
the speed work.
"""

import io
import math
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fast_scoring
from wordle import Game, GameState, Score
from wordlists import parse_file

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_score_cache(tmp_path, monkeypatch):
    """Keep Game's round-0 score-matrix caching out of the real
    .wordle_cache/ dir for every test in this module, so running the suite
    doesn't litter the project cache with entries for throwaway toy word
    lists. TestRealWordList overrides this with a no-op, since exercising
    the real cache is the whole point of that test.
    """
    monkeypatch.setattr(fast_scoring, "CACHE_DIR", tmp_path / "wordle_cache")


# ---------------------------------------------------------------------------
# get_score: core Wordle scoring semantics
# ---------------------------------------------------------------------------


class TestGetScore:
    def test_exact_match_is_all_green(self):
        g = Game(["apple"], ["apple"])
        assert g.get_score_list(g.get_score("apple", "apple")) == [Score.GREEN] * 5

    def test_no_shared_letters_is_all_gray(self):
        g = Game(["mount"], ["mount"])
        assert g.get_score_list(g.get_score("fjzkx", "mount")) == [Score.GRAY] * 5

    def test_duplicate_letter_in_guess_single_in_target(self):
        # Classic Wordle duplicate-letter case: guess has two Bs, target has
        # two Bs but only one lines up positionally with the correct match
        # already claimed by the green pass.
        g = Game(["bobby"], ["abbey"])
        assert g.get_score_list(g.get_score("bobby", "abbey")) == [
            Score.YELLOW,  # B: target has a B left over -> yellow
            Score.GRAY,    # O: not in target
            Score.GREEN,   # B: matches position 2
            Score.GRAY,    # B: target's B count already exhausted
            Score.GREEN,   # Y: matches position 4
        ]

    def test_both_duplicate_letters_get_yellow_when_target_has_both(self):
        # Target has two Es, guess has two Es in the wrong spots -> both
        # should be credited, not just the first.
        g = Game(["erase"], ["speed"])
        assert g.get_score_list(g.get_score("erase", "speed")) == [
            Score.YELLOW,  # E
            Score.GRAY,    # R
            Score.GRAY,    # A
            Score.YELLOW,  # S
            Score.YELLOW,  # E
        ]

    def test_mismatched_guess_target_length_raises(self):
        g = Game(["apple"], ["apple"])
        with pytest.raises(ValueError):
            g.get_score("appl", "apple")

    def test_guess_length_not_matching_word_length_raises(self):
        g = Game(["apple"], ["apple"])
        with pytest.raises(ValueError):
            g.get_score("abcd", "abcd")


# ---------------------------------------------------------------------------
# Score encoding: packed base-3 int <-> list <-> emoji string
# ---------------------------------------------------------------------------


class TestScoreEncoding:
    def test_all_gray_is_zero(self):
        g = Game(["apple"], ["apple"])
        assert g.get_score_num([Score.GRAY] * 5) == 0

    def test_all_green_is_max_value(self):
        g = Game(["apple"], ["apple"])
        assert g.get_score_num([Score.GREEN] * 5) == 3**5 - 1

    @pytest.mark.parametrize(
        "score_list",
        [
            [Score.GRAY] * 5,
            [Score.GREEN] * 5,
            [Score.GREEN, Score.GRAY, Score.YELLOW, Score.GRAY, Score.GREEN],
            [Score.YELLOW, Score.YELLOW, Score.YELLOW, Score.YELLOW, Score.YELLOW],
            [Score.GRAY, Score.GRAY, Score.GRAY, Score.GRAY, Score.GREEN],
        ],
    )
    def test_round_trip(self, score_list):
        g = Game(["apple"], ["apple"])
        num = g.get_score_num(score_list)
        assert g.get_score_list(num) == score_list

    def test_score_str_renders_expected_emoji(self):
        g = Game(["apple"], ["apple"])
        score_list = [Score.GRAY, Score.YELLOW, Score.GREEN, Score.GRAY, Score.GREEN]
        assert g.get_score_str(score_list) == "⬛🟨🟩⬛🟩"

    def test_score_str_accepts_packed_int(self):
        g = Game(["apple"], ["apple"])
        num = g.get_score_num(
            [Score.GRAY, Score.YELLOW, Score.GREEN, Score.GRAY, Score.GREEN]
        )
        assert g.get_score_str(num) == "⬛🟨🟩⬛🟩"


# ---------------------------------------------------------------------------
# Census / entropy, using a small hand-verifiable 2-letter word space
# ---------------------------------------------------------------------------


class TestCensusAndEntropy:
    def test_get_census_bincounts_scores(self):
        g = Game(["aa"], ["aa"])
        census = g.get_census(np.array([0, 2, 2, 8]))
        assert census.shape == (3**2,)
        assert census[0] == 1
        assert census[2] == 2
        assert census[8] == 1
        assert census.sum() == 4

    def test_guess_that_perfectly_splits_targets_has_max_entropy(self):
        # "aa" scores [aa, ab, ac] as [GG, G-, G-] with a distinct value per
        # bucket shape (1 unique, 2 tied) -> entropy = H([1/3, 2/3]).
        g = Game(["aa"], ["aa", "ab", "ac"])
        censuses = g.get_all_censuses()
        entropy = g.get_all_entropy(censuses)[0]
        expected = -(1 / 3 * math.log2(1 / 3) + 2 / 3 * math.log2(2 / 3))
        assert entropy == pytest.approx(expected)

    def test_guess_that_reveals_nothing_has_zero_entropy(self):
        # "x" never appears in any target, so every target scores identically
        # (green on the shared "a", gray on the uninformative second letter).
        g = Game(["ax"], ["aa", "ab", "ac"])
        censuses = g.get_all_censuses()
        entropy = g.get_all_entropy(censuses)[0]
        assert entropy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# find_best_guess: strategy selection + candidate-solution tie-break
# ---------------------------------------------------------------------------


class TestFindBestGuess:
    def test_picks_the_higher_entropy_guess(self):
        g = Game(["ax", "aa"], ["aa", "ab", "ac"])
        assert g.find_best_guess() == "aa"

    def test_tie_break_prefers_a_possible_solution(self):
        # "aa" and "bb" produce identical entropy against these targets (same
        # [1, 2]-shaped bucket split, just different score values). Only
        # "aa" is an actual candidate solution, so it should always win the
        # tie regardless of how np.argsort orders the tied pair.
        g = Game(["bb", "aa"], ["aa", "ab", "ac"])
        assert g.find_best_guess() == "aa"

    def test_num_top_guesses_does_not_change_the_winner(self):
        g = Game(["ax", "aa"], ["aa", "ab", "ac"], num_top_guesses=2)
        assert g.find_best_guess() == "aa"


# ---------------------------------------------------------------------------
# get_guess_score: solution-driven, manual entry, and command handling
# ---------------------------------------------------------------------------


class TestGetGuessScore:
    def test_known_solution_ignores_potential_score(self):
        g = Game(["aa"], ["aa", "bb"], solution="bb")
        state, score = g.get_guess_score("aa", "garbage")
        assert state == GameState.CONTINUE
        assert score == g.get_score("aa", "bb")

    def test_potential_score_wrong_length_is_ignored(self):
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input", return_value="01"):
            state, score = g.get_guess_score("aa", "1")
        assert state == GameState.CONTINUE
        assert score == 1  # int("01", base=3)

    def test_potential_score_valid_is_used_without_prompting(self):
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input") as mock_input:
            state, score = g.get_guess_score("aa", "12")
        mock_input.assert_not_called()
        assert state == GameState.CONTINUE
        assert score == int("12", base=3)

    def test_potential_score_invalid_digits_falls_back_to_prompt(self):
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input", return_value="01"):
            state, score = g.get_guess_score("aa", "1a")
        assert state == GameState.CONTINUE
        assert score == 1

    def test_restart_command(self):
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input", return_value="restart"):
            state, score = g.get_guess_score("aa")
        assert state == GameState.RESTART
        assert score is None

    def test_quit_command(self):
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input", return_value="quit"):
            state, score = g.get_guess_score("aa")
        assert state == GameState.QUIT
        assert score is None

    def test_eof_quits(self):
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input", side_effect=EOFError):
            state, score = g.get_guess_score("aa")
        assert state == GameState.QUIT
        assert score is None


# ---------------------------------------------------------------------------
# play_one_round / reset: the full interactive state machine
# ---------------------------------------------------------------------------


class TestPlayOneRound:
    def test_automatic_solve_deduces_unique_remaining_candidate(self):
        g = Game(
            ["aa", "ab", "ba", "bb"],
            ["aa", "ab", "ba", "bb"],
            solution="bb",
            automatic=True,
            initial_guess="aa",
        )
        state = g.play_one_round()
        assert state == GameState.SOLVED
        assert g.found_solution == "bb"
        # One real guess ("aa") plus a synthesized perfect-score entry for
        # the deduced answer, since "aa" alone didn't score all-green.
        assert g._scores == [("aa", 0), ("bb", g.get_score_num([Score.GREEN] * 2))]

    def test_guessing_the_answer_directly_does_not_duplicate_score_entry(self):
        g = Game(
            ["aa", "bb"], ["aa", "bb"], solution="bb", automatic=True, initial_guess="bb"
        )
        state = g.play_one_round()
        assert state == GameState.SOLVED
        assert g.found_solution == "bb"
        assert g._scores == [("bb", g.get_score_num([Score.GREEN] * 2))]

    def test_no_candidates_match_score_is_an_error(self):
        # guess "aa" can only ever score 8 (vs "aa") or 0 (vs "bb"); "01" (1)
        # is a well-formed but unachievable score for this pool.
        g = Game(["aa"], ["aa", "bb"])
        with patch("builtins.input", return_value="01"):
            state = g.play_one_round()
        assert state == GameState.ERROR

    def test_reset_restores_initial_state(self):
        g = Game(
            ["aa", "bb"], ["aa", "bb"], solution="bb", automatic=True, initial_guess="bb"
        )
        g.play_one_round()
        assert g.found_solution is not None
        g.reset()
        assert g.found_solution is None
        assert g._scores == []
        assert len(g.target_lists) == 1
        assert list(g.target_lists[0]) == ["aa", "bb"]

    def test_already_solved_round_is_a_noop_quit(self):
        g = Game(
            ["aa", "bb"], ["aa", "bb"], solution="bb", automatic=True, initial_guess="bb"
        )
        g.play_one_round()
        assert g.play_one_round() == GameState.QUIT

    def test_interactive_guess_override_replaces_suggestion(self):
        g = Game(
            ["aa", "bb"], ["aa", "bb"], solution="bb", initial_guess="aa"
        )
        with patch("builtins.input", return_value="bb"):
            state = g.play_one_round()
        assert state == GameState.SOLVED
        assert g.found_solution == "bb"
        assert g._scores == [("bb", g.get_score_num([Score.GREEN] * 2))]

    def test_interactive_eof_quits(self):
        g = Game(["aa"], ["aa", "bb"], initial_guess="aa")
        with patch("builtins.input", side_effect=EOFError):
            assert g.play_one_round() == GameState.QUIT


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_basic_parse(self):
        wl = parse_file(io.StringIO("aa\nbb\n\ncc\ndd\n"))
        assert wl.target == ["aa", "bb"]
        assert wl.extra == ["cc", "dd"]
        assert wl.word_length == 2

    def test_no_extra_section(self):
        wl = parse_file(io.StringIO("aa\nbb\n"))
        assert wl.target == ["aa", "bb"]
        assert wl.extra == []
        assert wl.word_length == 2

    def test_words_without_a_weight_column_default_to_uniform_weight(self):
        wl = parse_file(io.StringIO("aa\nbb\n"))
        assert wl.weights == {"aa": 1.0, "bb": 1.0}

    def test_trailing_weight_column_is_parsed(self):
        wl = parse_file(io.StringIO("aa 58\nbb 52.8829\n"))
        assert wl.target == ["aa", "bb"]
        assert wl.word_length == 2
        assert wl.weights == {"aa": 58.0, "bb": 52.8829}

    def test_length_check_uses_word_not_full_line(self):
        # "bb 123456789" is a long line, but the word itself is still 2
        # chars -- the weight column must not trip length validation.
        wl = parse_file(io.StringIO("aa 58\nbb 123456789\n"))
        assert wl.target == ["aa", "bb"]
        assert wl.word_length == 2

    def test_overlap_between_target_and_extra_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            parse_file(io.StringIO("aa\nbb\n\nbb\ncc\n"))

    def test_length_mismatch_raises_with_interpolated_values(self):
        with pytest.raises(ValueError, match=r"Bad length 3, expected 2"):
            parse_file(io.StringIO("aa\nbbb\n"))

    def test_too_many_blank_lines_raises(self):
        with pytest.raises(ValueError, match="too many blank lines"):
            parse_file(io.StringIO("aa\n\ncc\n\ndd\n"))

    def test_real_wordle_wordlist(self):
        with open(REPO_ROOT / "words.wordle.txt") as fp:
            wl = parse_file(fp)
        assert wl.word_length == 5
        assert len(wl.target) == 2309
        assert len(wl.extra) == 12546
        assert not set(wl.target) & set(wl.extra)
        assert all(w == 1.0 for w in wl.weights.values())

    def test_real_weighted_wordlist(self):
        # "word <weight>" per line, no separate extra section -- the same
        # list is meant to serve as both guesses and targets.
        with open(REPO_ROOT / "words.weighted.txt") as fp:
            wl = parse_file(fp)
        assert wl.word_length == 5
        assert len(wl.target) == 3209
        assert wl.extra == []
        assert all(word.isalpha() and word.islower() for word in wl.target)
        assert set(wl.weights) == set(wl.target)
        assert all(isinstance(w, float) for w in wl.weights.values())


# ---------------------------------------------------------------------------
# Slow: regression baseline against the real word lists, and speed benchmark
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def word_data():
    with open(REPO_ROOT / "words.wordle.txt") as fp:
        wl = parse_file(fp)
    words = sorted(set(wl.target) | set(wl.extra))
    return words, wl.target


@pytest.mark.slow
class TestRealWordList:
    @pytest.fixture(autouse=True)
    def isolated_score_cache(self):
        # Overrides the module-level isolation fixture with a no-op: this
        # test's job is to exercise (and warm) the real on-disk cache.
        yield

    def test_round_one_best_guess_is_a_known_golden_value(self, word_data, caplog):
        # A single find_best_guess() call is the expensive operation this
        # test exists to pin down; the entropy value is pulled from its
        # log line rather than recomputed, to avoid paying that cost twice.
        guesses, targets = word_data
        g = Game(guesses, targets)
        with caplog.at_level("INFO"):
            best_guess = g.find_best_guess()
        assert best_guess == "tarse"

        match = re.search(r"Best guess \S+ entropy (\S+)", caplog.text)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(5.948974509955522)


# ---------------------------------------------------------------------------
# fast_scoring: vectorized score_matrix must agree with Game.get_score
# exactly, and the on-disk cache must actually short-circuit recomputation.
# ---------------------------------------------------------------------------


class TestScoreMatrixMatchesScalar:
    @pytest.mark.parametrize(
        "guess, target",
        [
            ("apple", "apple"),
            ("fjzkx", "mount"),
            ("bobby", "abbey"),
            ("erase", "speed"),
        ],
    )
    def test_single_pair(self, guess, target):
        g = Game([guess], [target])
        assert fast_scoring.score_matrix([guess], [target])[0, 0] == g.get_score(
            guess, target
        )

    def test_cross_check_against_scalar_over_many_pairs(self):
        # A broad, randomized cross-check over real 5-letter words -- the
        # hand-picked pairs above cover the known tricky duplicate-letter
        # cases, but the vectorized algorithm's correctness claim is really
        # "matches the scalar implementation for every pair", so verify that
        # directly rather than trusting a handful of examples.
        with open(REPO_ROOT / "words.wordle.txt") as fp:
            wl = parse_file(fp)
        words = sorted(set(wl.target) | set(wl.extra))

        import random

        rng = random.Random(0)
        guesses = rng.sample(words, 20)
        targets = rng.sample(words, 20)

        g = Game(guesses, targets)
        matrix = fast_scoring.score_matrix(guesses, targets)
        for gi, guess in enumerate(guesses):
            for ti, tgt in enumerate(targets):
                assert matrix[gi, ti] == g.get_score(guess, tgt), (guess, tgt)

    def test_game_score_guess_matches_scalar_loop(self):
        g = Game(["bobby"], ["abbey", "speed", "erase", "bobby"])
        vectorized = g.score_guess("bobby")
        scalar = [g.get_score("bobby", t) for t in ["abbey", "speed", "erase", "bobby"]]
        assert list(vectorized) == scalar


class TestScoreMatrixCache:
    def test_cache_hit_skips_recomputation(self, tmp_path):
        cache_dir = tmp_path / "cache"
        with patch(
            "fast_scoring.score_matrix", wraps=fast_scoring.score_matrix
        ) as spy:
            first = fast_scoring.cached_score_matrix(
                ["aa", "ab"], ["aa", "ab", "ba"], cache_dir=cache_dir
            )
            second = fast_scoring.cached_score_matrix(
                ["aa", "ab"], ["aa", "ab", "ba"], cache_dir=cache_dir
            )
        spy.assert_called_once()
        np.testing.assert_array_equal(first, second)

    def test_different_word_lists_produce_different_cache_entries(self, tmp_path):
        cache_dir = tmp_path / "cache"
        fast_scoring.cached_score_matrix(["aa"], ["aa", "ab"], cache_dir=cache_dir)
        fast_scoring.cached_score_matrix(["aa"], ["aa", "bb"], cache_dir=cache_dir)
        assert len(list(cache_dir.glob("*.npy"))) == 2

    def test_cached_result_matches_uncached(self, tmp_path):
        cache_dir = tmp_path / "cache"
        guesses, targets = ["bobby", "abbey"], ["abbey", "speed", "erase"]
        cached = fast_scoring.cached_score_matrix(guesses, targets, cache_dir=cache_dir)
        uncached = fast_scoring.score_matrix(guesses, targets)
        np.testing.assert_array_equal(cached, uncached)

    def test_round_zero_populates_the_cache(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(fast_scoring, "CACHE_DIR", cache_dir)
        g = Game(["aa", "ab", "ba", "bb"], ["aa", "ab", "ba", "bb"])
        g.get_all_censuses()
        assert len(list(cache_dir.glob("*.npy"))) == 1
