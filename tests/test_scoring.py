"""Tests for scoring.py's scalar (guess, target) scoring and packed-int
encode/decode.

Ported from the old Game-based tests (which exercised this exact algorithm
through Game.get_score/get_score_num/get_score_list/get_score_str) now that
scoring is a pure module with no Game to attach the tests to.
"""

import pytest

from scoring import Score, get_score, get_score_list, get_score_num, get_score_str


class TestGetScore:
    def test_exact_match_is_all_green(self):
        assert get_score_list(get_score("apple", "apple"), 5) == [Score.GREEN] * 5

    def test_no_shared_letters_is_all_gray(self):
        assert get_score_list(get_score("fjzkx", "mount"), 5) == [Score.GRAY] * 5

    def test_duplicate_letter_in_guess_single_in_target(self):
        # Classic Wordle duplicate-letter case: guess has two Bs, target has
        # two Bs but only one lines up positionally with the correct match
        # already claimed by the green pass.
        assert get_score_list(get_score("bobby", "abbey"), 5) == [
            Score.YELLOW,  # B: target has a B left over -> yellow
            Score.GRAY,    # O: not in target
            Score.GREEN,   # B: matches position 2
            Score.GRAY,    # B: target's B count already exhausted
            Score.GREEN,   # Y: matches position 4
        ]

    def test_both_duplicate_letters_get_yellow_when_target_has_both(self):
        # Target has two Es, guess has two Es in the wrong spots -> both
        # should be credited, not just the first.
        assert get_score_list(get_score("erase", "speed"), 5) == [
            Score.YELLOW,  # E
            Score.GRAY,    # R
            Score.GRAY,    # A
            Score.YELLOW,  # S
            Score.YELLOW,  # E
        ]

    def test_mismatched_guess_target_length_raises(self):
        with pytest.raises(ValueError):
            get_score("appl", "apple")


class TestScoreEncoding:
    def test_all_gray_is_zero(self):
        assert get_score_num([Score.GRAY] * 5) == 0

    def test_all_green_is_max_value(self):
        assert get_score_num([Score.GREEN] * 5) == 3**5 - 1

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
        num = get_score_num(score_list)
        assert get_score_list(num, len(score_list)) == score_list

    def test_score_str_renders_expected_emoji(self):
        score_list = [Score.GRAY, Score.YELLOW, Score.GREEN, Score.GRAY, Score.GREEN]
        assert get_score_str(score_list) == "⬛🟨🟩⬛🟩"

    def test_score_str_accepts_packed_int(self):
        num = get_score_num(
            [Score.GRAY, Score.YELLOW, Score.GREEN, Score.GRAY, Score.GREEN]
        )
        assert get_score_str(num, 5) == "⬛🟨🟩⬛🟩"

    def test_score_str_packed_int_without_n_raises(self):
        with pytest.raises(ValueError):
            get_score_str(0)
