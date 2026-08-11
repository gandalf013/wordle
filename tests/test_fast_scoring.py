"""Tests for fast_scoring.py: vectorized score_matrix must agree with
scoring.get_score exactly, and the on-disk cache must actually
short-circuit recomputation.
"""

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fast_scoring
from scoring import get_score
from wordlists import parse_file

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_score_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fast_scoring, "CACHE_DIR", tmp_path / "wordle_cache")


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
        assert fast_scoring.score_matrix([guess], [target])[0, 0] == get_score(
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

        rng = random.Random(0)
        guesses = rng.sample(words, 20)
        targets = rng.sample(words, 20)

        matrix = fast_scoring.score_matrix(guesses, targets)
        for gi, guess in enumerate(guesses):
            for ti, tgt in enumerate(targets):
                assert matrix[gi, ti] == get_score(guess, tgt), (guess, tgt)


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
        fast_scoring.cached_score_matrix(
            ["aa", "ab", "ba", "bb"], ["aa", "ab", "ba", "bb"], cache_dir=cache_dir
        )
        assert len(list(cache_dir.glob("*.npy"))) == 1

    def test_show_progress_tqdm_invocation(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        guesses = [
            f"{alphabet[i % 26]}{alphabet[(i // 26) % 26]}{alphabet[(i // 676) % 26]}aa"
            for i in range(1500)
        ]
        targets = ["aaaaa", "bbbbb"]
        with patch("fast_scoring.tqdm", wraps=fast_scoring.tqdm) as spy:
            fast_scoring.score_matrix(guesses, targets, show_progress=True)
            spy.assert_called_once()

        with patch("fast_scoring.tqdm", wraps=fast_scoring.tqdm) as spy:
            fast_scoring.score_matrix(guesses, targets, show_progress=False)
            spy.assert_not_called()


class TestScoreMatrixAndBincounts:
    def test_fused_matches_separate_matrix_and_bincounts(self):
        guesses = ["bobby", "erase", "speed"]
        targets = ["abbey", "speed", "erase", "mount"]
        weights = {"abbey": 1.0, "speed": 2.5, "erase": 0.5, "mount": 1.2}

        matrix_sep = fast_scoring.score_matrix(guesses, targets)
        w_arr = np.array([weights[w] for w in targets], dtype=np.float64)
        counts_sep, masses_sep = fast_scoring.bincount_scores(matrix_sep, weights=w_arr)

        matrix_fused, counts_fused, masses_fused = fast_scoring.score_matrix_and_bincounts(
            guesses, targets, weights=weights
        )

        np.testing.assert_array_equal(matrix_fused, matrix_sep)
        np.testing.assert_allclose(counts_fused, counts_sep)
        np.testing.assert_allclose(masses_fused, masses_sep)

    def test_numpy_fallback_matches_c_implementation(self, monkeypatch):
        guesses = ["bobby", "erase", "speed", "apple", "fjzkx"]
        targets = ["abbey", "speed", "erase", "mount", "apple"]
        weights = {"abbey": 1.0, "speed": 2.5, "erase": 0.5, "mount": 1.2, "apple": 0.8}

        mat_c = fast_scoring._score_matrix_c(guesses, targets)
        mat_np = fast_scoring._score_matrix_numpy(guesses, targets)
        np.testing.assert_array_equal(mat_c, mat_np)

        monkeypatch.setattr(fast_scoring, "HAS_C_LIB", False)
        mat_fallback, counts_fallback, masses_fallback = fast_scoring.score_matrix_and_bincounts(
            guesses, targets, weights=weights
        )
        np.testing.assert_array_equal(mat_fallback, mat_c)

