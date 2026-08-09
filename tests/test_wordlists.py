"""Tests for wordlists.parse_file."""

import io
from pathlib import Path

import pytest

from wordlists import parse_file

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    def test_empty_file_raises(self):
        with pytest.raises(ValueError, match="Empty word list file"):
            parse_file(io.StringIO(""))

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
