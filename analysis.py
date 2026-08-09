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

    `buckets` is None when `analyze_all` is called (unless `include_buckets=True`),
    as strategies only require summary statistics for ranking. Single-guess
    `analyze` and display helpers populate `buckets`.

    `bucket_counts`/`bucket_masses` are the non-zero score buckets as compact
    (score, count) / (score, mass) pairs -- what strategies like
    TwoPlyExpectimaxStrategy need without carrying the full word lists.
    `analyze` always populates them (masses only when weights are supplied);
    `analyze_all` populates them only when `include_bucket_stats=True`.

    The weighted_* fields are None when no weights were supplied to
    analyze()/analyze_all() -- callers must not assume they're populated.
    """

    guess: str
    entropy: float  # bits, uniform over bucket counts
    worst_case_size: int  # size of the largest bucket
    expected_size: float  # sum(p_i * bucket_size_i), uniform
    is_possible_solution: bool
    buckets: dict[int, list[str]] | None = None

    weighted_entropy: float | None = None
    weighted_expected_size: float | None = None
    solution_probability: float | None = None

    bucket_counts: tuple[tuple[int, int], ...] | None = None
    bucket_masses: tuple[tuple[int, float], ...] | None = None


def _buckets_from_scores(
    target_pool: Sequence[str], scores: Sequence[int]
) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {}
    for target, score in zip(target_pool, scores):
        buckets.setdefault(int(score), []).append(target)
    return buckets


def bucket_counts_from_buckets(
    buckets: dict[int, list[str]],
) -> tuple[tuple[int, int], ...]:
    """Compact (score, count) pairs for each non-empty bucket, sorted by
    score. The single source of truth for deriving GuessAnalysis.bucket_counts
    from a raw `buckets` dict, used both by `analyze`/`analyze_all` and by
    strategies that fall back to `.buckets` when `.bucket_counts` isn't set."""
    return tuple(sorted((int(s), len(words)) for s, words in buckets.items()))


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

    bucket_counts = bucket_counts_from_buckets(buckets)
    bucket_masses = None
    weighted_entropy = weighted_expected_size = solution_probability = None

    if weights is not None:
        # Compute bucket masses once to share between bucket_masses tuple and masses array
        bucket_masses_dict = {
            int(s): sum(weights.get(w, 1.0) for w in words)
            for s, words in buckets.items()
        }
        bucket_masses = tuple(sorted(bucket_masses_dict.items()))

        masses = np.array(list(bucket_masses_dict.values()), dtype=np.float64)
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
        bucket_counts=bucket_counts,
        bucket_masses=bucket_masses,
    )


def analyze(
    guess: str,
    target_pool: Sequence[str],
    weights: dict[str, float] | None = None,
    use_cache: bool = False,
    show_progress: bool | None = None,
) -> GuessAnalysis:
    """Score `guess` against every word in `target_pool` (via
    fast_scoring.score_matrix, not a fresh Python loop) and summarize the
    split.

    Single entry point used by:
      - strategies, to rank candidate guesses
      - the `analyze <word>` REPL command (peek without committing)
      - the `buckets <word>` REPL command (renders `.buckets` directly)

    `use_cache=True` persists the (1, T) score matrix to the on-disk cache,
    so repeated peeks at the same word/pool pair (e.g. the REPL `analyze`
    and `buckets` commands across restarts) load instantly instead of
    re-scoring.
    """
    target_pool = list(target_pool)
    scorer = fast_scoring.cached_score_matrix if use_cache else fast_scoring.score_matrix
    scores = scorer([guess], target_pool, show_progress=show_progress)[0]
    buckets = _buckets_from_scores(target_pool, scores)
    return _analysis_from_buckets(guess, buckets, frozenset(target_pool), weights)


def analyze_all(
    guess_list: Sequence[str],
    target_pool: Sequence[str],
    weights: dict[str, float] | None = None,
    use_cache: bool = False,
    include_buckets: bool = False,
    include_bucket_stats: bool = False,
    show_progress: bool | None = None,
) -> list[GuessAnalysis]:
    """analyze() for every candidate guess, backed by vectorized bincount stats
    and optional score_matrix caching.

    By default (`include_buckets=False`), `buckets` is set to None on each
    `GuessAnalysis` to avoid constructing millions of Python dictionary
    entries during ranking. `include_bucket_stats=True` additionally
    populates `bucket_counts`/`bucket_masses` (the compact per-guess bucket
    tallies) -- also vectorized, and only needed by strategies like
    TwoPlyExpectimaxStrategy, so it's opt-in too.
    """
    guess_list = list(guess_list)
    target_pool = list(target_pool)
    target_set = frozenset(target_pool)
    G = len(guess_list)
    T = len(target_pool)
    if not G or not T:
        return []

    scorer = fast_scoring.cached_score_matrix if use_cache else fast_scoring.score_matrix
    matrix = scorer(guess_list, target_pool, show_progress=show_progress)

    target_weights = None
    if weights is not None:
        target_weights = np.array([weights.get(w, 1.0) for w in target_pool], dtype=np.float64)
    counts, masses = fast_scoring.bincount_scores(matrix, weights=target_weights)

    bucket_counts_list: list[tuple[tuple[int, int], ...] | None] = [None] * G
    bucket_masses_list: list[tuple[tuple[int, float], ...] | None] = [None] * G
    if include_bucket_stats:
        for i in range(G):
            nz = np.flatnonzero(counts[i])
            c_nz = counts[i][nz].tolist()
            bucket_counts_list[i] = tuple(zip(nz.tolist(), c_nz))
            if weights is not None:
                m_nz = masses[i][nz].tolist()
                bucket_masses_list[i] = tuple(zip(nz.tolist(), m_nz))

    probs = counts / T
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -np.sum(probs * log_probs, axis=1)
    worst_case_size = np.max(counts, axis=1).astype(int)
    expected_size = np.sum(counts**2, axis=1) / T

    entropy_list = entropy.tolist()
    worst_case_list = worst_case_size.tolist()
    expected_size_list = expected_size.tolist()

    if weights is not None:
        total_masses = np.sum(masses, axis=1)
        w_probs = np.where(total_masses[:, None] > 0, masses / total_masses[:, None], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_log_probs = np.where(w_probs > 0, np.log2(w_probs), 0.0)
            weighted_entropy = -np.sum(w_probs * w_log_probs, axis=1)
        weighted_expected_size = np.sum(w_probs * counts, axis=1)

        w_entropy_list = weighted_entropy.tolist()
        w_expected_list = weighted_expected_size.tolist()
        total_masses_list = total_masses.tolist()
    else:
        w_entropy_list = None
        w_expected_list = None
        total_masses_list = None

    analyses = []
    for i, guess in enumerate(guess_list):
        is_possible_solution = guess in target_set
        w_ent = w_entropy_list[i] if w_entropy_list is not None else None
        w_exp = w_expected_list[i] if w_expected_list is not None else None
        sol_prob = None
        if total_masses_list is not None:
            tm = total_masses_list[i]
            g_w = weights.get(guess, 1.0) if is_possible_solution else 0.0
            sol_prob = g_w / tm if tm else 0.0

        buckets = _buckets_from_scores(target_pool, matrix[i]) if include_buckets else None

        analyses.append(
            GuessAnalysis(
                guess=guess,
                buckets=buckets,
                entropy=entropy_list[i],
                worst_case_size=worst_case_list[i],
                expected_size=expected_size_list[i],
                is_possible_solution=is_possible_solution,
                weighted_entropy=w_ent,
                weighted_expected_size=w_exp,
                solution_probability=sol_prob,
                bucket_counts=bucket_counts_list[i],
                bucket_masses=bucket_masses_list[i],
            )
        )
    return analyses
