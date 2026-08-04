"""Guess analysis: buckets, entropy, and other stats derived from scoring a
candidate guess against a target pool.

This module never scores a (guess, target) pair itself -- it aggregates an
existing fast_scoring score matrix into per-guess summaries that strategies
and display code can read off directly instead of each recomputing scores
independently.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import entropy as get_entropy

import fast_scoring


@dataclass(frozen=True)
class GuessAnalysis:
    """Everything derived from scoring one candidate guess against a target
    pool. Computed once per (guess, pool[, weights]) triple; every strategy
    and every display view reads off this instead of each recomputing
    scores independently.

    The weighted_* fields are None when no weights were supplied to
    analyze()/analyze_all() -- callers must not assume they're populated.
    """

    guess: str
    buckets: dict[int, list[str]]  # packed score -> matching words
    entropy: float  # bits, uniform over bucket counts
    worst_case_size: int  # size of the largest bucket
    expected_size: float  # sum(p_i * bucket_size_i), uniform
    is_possible_solution: bool

    weighted_entropy: float | None = None
    weighted_expected_size: float | None = None
    solution_probability: float | None = None


def _buckets_from_scores(
    target_pool: Sequence[str], scores: Sequence[int]
) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {}
    for target, score in zip(target_pool, scores):
        buckets.setdefault(int(score), []).append(target)
    return buckets


def _analysis_from_buckets(
    guess: str,
    buckets: dict[int, list[str]],
    target_set: frozenset[str],
    weights: dict[str, float] | None,
) -> GuessAnalysis:
    sizes = np.array([len(words) for words in buckets.values()], dtype=np.int64)
    total = int(sizes.sum())
    entropy = float(get_entropy(sizes, base=2)) if total else 0.0
    worst_case_size = int(sizes.max()) if sizes.size else 0
    expected_size = float((sizes * sizes).sum() / total) if total else 0.0
    is_possible_solution = guess in target_set

    weighted_entropy = weighted_expected_size = solution_probability = None
    if weights is not None:
        # Missing entries default to 1.0 (uniform), matching WordList's own
        # default -- callers may pass a weights dict that doesn't cover
        # every word in target_pool (e.g. an "extra" guess-only word).
        masses = np.array(
            [sum(weights.get(w, 1.0) for w in words) for words in buckets.values()],
            dtype=np.float64,
        )
        total_mass = float(masses.sum())
        if total_mass:
            weighted_entropy = float(get_entropy(masses, base=2))
            weighted_expected_size = float((masses / total_mass * sizes).sum())
        else:
            weighted_entropy = 0.0
            weighted_expected_size = 0.0

        # A guess that isn't itself a candidate can't be the answer,
        # regardless of what a weights dict happens to say about it.
        guess_weight = weights.get(guess, 1.0) if is_possible_solution else 0.0
        solution_probability = guess_weight / total_mass if total_mass else 0.0

    return GuessAnalysis(
        guess=guess,
        buckets=buckets,
        entropy=entropy,
        worst_case_size=worst_case_size,
        expected_size=expected_size,
        is_possible_solution=is_possible_solution,
        weighted_entropy=weighted_entropy,
        weighted_expected_size=weighted_expected_size,
        solution_probability=solution_probability,
    )


def analyze(
    guess: str,
    target_pool: Sequence[str],
    weights: dict[str, float] | None = None,
) -> GuessAnalysis:
    """Score `guess` against every word in `target_pool` (via
    fast_scoring.score_matrix, not a fresh Python loop) and summarize the
    split.

    Single entry point used by:
      - strategies, to rank candidate guesses
      - the `analyze <word>` REPL command (peek without committing)
      - the `buckets <word>` REPL command (renders `.buckets` directly)
    """
    target_pool = list(target_pool)
    scores = fast_scoring.score_matrix([guess], target_pool)[0]
    buckets = _buckets_from_scores(target_pool, scores)
    return _analysis_from_buckets(guess, buckets, frozenset(target_pool), weights)


def analyze_all(
    guess_list: Sequence[str],
    target_pool: Sequence[str],
    weights: dict[str, float] | None = None,
    use_cache: bool = False,
) -> list[GuessAnalysis]:
    """analyze() for every candidate guess, backed by a single
    fast_scoring score_matrix call rather than one matrix build per guess --
    this is the direct replacement for today's Game.get_all_censuses.

    `use_cache=True` routes through fast_scoring.cached_score_matrix instead
    of score_matrix -- callers should only set this for the expensive
    round-1-against-the-full-list case (mirroring Game.get_all_censuses'
    `self.round == 0` check), since the on-disk cache is keyed on the exact
    ordered word lists and every narrower round would otherwise mint its own
    one-off cache entry.
    """
    guess_list = list(guess_list)
    target_pool = list(target_pool)
    target_set = frozenset(target_pool)

    scorer = fast_scoring.cached_score_matrix if use_cache else fast_scoring.score_matrix
    matrix = scorer(guess_list, target_pool)

    analyses = []
    for i, guess in enumerate(guess_list):
        buckets = _buckets_from_scores(target_pool, matrix[i])
        analyses.append(_analysis_from_buckets(guess, buckets, target_set, weights))
    return analyses
