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


def _reference_stats(guesses, targets, weights):
    """Independent reimplementation of the entropy/expected-size reduction
    (the same math analyze_all used before score_and_analyze existed), used
    as ground truth for the fused C path and the NumPy fallback alike."""
    matrix, counts, masses = fast_scoring.score_matrix_and_bincounts(
        guesses, targets, weights=weights
    )
    T = len(targets)
    probs = counts / T
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -np.sum(probs * log_probs, axis=1)
    worst_case_size = np.max(counts, axis=1)
    expected_size = np.sum(counts**2, axis=1) / T
    if weights is None:
        return matrix, counts, masses, entropy, worst_case_size, expected_size, None, None, None
    total_mass = np.sum(masses, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_probs = np.where(total_mass[:, None] > 0, masses / total_mass[:, None], 0.0)
        w_log_probs = np.where(w_probs > 0, np.log2(w_probs), 0.0)
        weighted_entropy = -np.sum(w_probs * w_log_probs, axis=1)
    weighted_expected_size = np.sum(w_probs * counts, axis=1)
    return (
        matrix, counts, masses, entropy, worst_case_size, expected_size,
        weighted_entropy, weighted_expected_size, total_mass,
    )


class TestScoreAndAnalyze:
    """score_and_analyze fuses the entropy/worst-case/expected-size
    reduction into the same C pass that builds the bucket counts, instead of
    analyze_all doing several NumPy passes over a (G, 243) array. These
    tests pin its output to an independent NumPy reimplementation of that
    same math, for both the C path and the (HAS_C_LIB=False) fallback."""

    GUESSES = ["bobby", "erase", "speed", "apple", "fjzkx", "mount"]
    TARGETS = ["abbey", "speed", "erase", "mount", "apple", "fjzkx"]
    WEIGHTS = {"abbey": 1.0, "speed": 2.5, "erase": 0.5, "mount": 1.2, "apple": 0.8, "fjzkx": 0.1}

    @pytest.mark.parametrize("weights", [None, WEIGHTS])
    def test_matches_reference_reduction(self, weights):
        ref = _reference_stats(self.GUESSES, self.TARGETS, weights)
        stats = fast_scoring.score_and_analyze(
            self.GUESSES, self.TARGETS, weights=weights,
            need_matrix=True, need_bucket_arrays=True,
        )

        np.testing.assert_array_equal(stats.matrix, ref[0])
        np.testing.assert_allclose(stats.counts, ref[1])
        np.testing.assert_allclose(stats.masses, ref[2])
        np.testing.assert_allclose(stats.entropy, ref[3], atol=1e-9)
        np.testing.assert_allclose(stats.worst_case_size, ref[4], atol=1e-9)
        np.testing.assert_allclose(stats.expected_size, ref[5], atol=1e-9)
        if weights is None:
            assert stats.weighted_entropy is None
            assert stats.weighted_expected_size is None
            assert stats.total_mass is None
        else:
            np.testing.assert_allclose(stats.weighted_entropy, ref[6], atol=1e-9)
            np.testing.assert_allclose(stats.weighted_expected_size, ref[7], atol=1e-9)
            np.testing.assert_allclose(stats.total_mass, ref[8], atol=1e-9)

    def test_numpy_fallback_matches_c_path(self, monkeypatch):
        c_stats = fast_scoring.score_and_analyze(
            self.GUESSES, self.TARGETS, weights=self.WEIGHTS,
            need_matrix=True, need_bucket_arrays=True,
        )
        monkeypatch.setattr(fast_scoring, "HAS_C_LIB", False)
        fallback_stats = fast_scoring.score_and_analyze(
            self.GUESSES, self.TARGETS, weights=self.WEIGHTS,
            need_matrix=True, need_bucket_arrays=True,
        )

        np.testing.assert_array_equal(fallback_stats.matrix, c_stats.matrix)
        np.testing.assert_allclose(fallback_stats.entropy, c_stats.entropy, atol=1e-9)
        np.testing.assert_allclose(fallback_stats.worst_case_size, c_stats.worst_case_size, atol=1e-9)
        np.testing.assert_allclose(fallback_stats.expected_size, c_stats.expected_size, atol=1e-9)
        np.testing.assert_allclose(fallback_stats.weighted_entropy, c_stats.weighted_entropy, atol=1e-9)
        np.testing.assert_allclose(
            fallback_stats.weighted_expected_size, c_stats.weighted_expected_size, atol=1e-9
        )
        np.testing.assert_allclose(fallback_stats.total_mass, c_stats.total_mass, atol=1e-9)

    def test_matrix_and_bucket_arrays_are_none_unless_requested(self):
        stats = fast_scoring.score_and_analyze(self.GUESSES, self.TARGETS, weights=self.WEIGHTS)
        assert stats.matrix is None
        assert stats.counts is None
        assert stats.masses is None
        assert stats.entropy is not None

    def test_use_cache_matches_uncached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fast_scoring, "CACHE_DIR", tmp_path / "wordle_cache")
        uncached = fast_scoring.score_and_analyze(self.GUESSES, self.TARGETS, weights=self.WEIGHTS)
        cached = fast_scoring.score_and_analyze(
            self.GUESSES, self.TARGETS, weights=self.WEIGHTS, use_cache=True
        )
        np.testing.assert_allclose(cached.entropy, uncached.entropy, atol=1e-9)
        np.testing.assert_allclose(cached.weighted_entropy, uncached.weighted_entropy, atol=1e-9)

    def test_empty_guesses_or_targets(self):
        stats = fast_scoring.score_and_analyze([], self.TARGETS, weights=self.WEIGHTS)
        assert stats.entropy.shape == (0,)
        stats = fast_scoring.score_and_analyze(self.GUESSES, [], weights=self.WEIGHTS)
        assert stats.entropy.shape == (len(self.GUESSES),)

