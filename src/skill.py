"""Wordlebot-style skill and luck scores.

Both are pure percentile-style functions over data `analysis.py` already
computes (a guess's entropy and its bucket sizes), so a web backend can
call them with no additional scoring. Nothing here does scoring itself and
nothing holds game state -- it mirrors the `strategies.py` convention.

Definitions (both map to 0-100):

skill
    How informative `guess` is relative to every other guess at the same
    state. This is the percentile rank of `guess.entropy` within the
    entropy of all guesses::

        skill = 100 * (#guesses with entropy <= guess.entropy) / total

    so the single most informative guess scores 100 and the least
    informative scores 100/N (approaching 0 as the pool grows).

luck
    How much the *actual* pattern narrowed the pool relative to what this
    guess could have achieved. Measured in bits actually revealed by the
    outcome, normalized between the guess's worst and best possible
    buckets::

        info(k)  = log2(N) - log2(k)   bits revealed by a bucket of size k
        luck     = 100 * (info(actual) - info(min)) / (info(max) - info(min))

    where info(min) is the guess's largest bucket (worst outcome) and
    info(max) is a single-survivor bucket (the answer). Landing in the
    guess's largest bucket scores 0; landing in a 1-word bucket scores 100.
    This is a linear-in-information normalization (a deliberate, easily
    swapped choice) rather than a probability-mass percentile, so "expected"
    outcomes land near the middle and lucky/unlucky tail outcomes are
    symmetric.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from analysis import GuessAnalysis


def skill_score(guess_entropy: float, all_entropies: Sequence[float]) -> float:
    """Percentile rank (0-100) of `guess_entropy` within `all_entropies`."""
    n = len(all_entropies)
    if n == 0:
        return 0.0
    count = sum(1 for e in all_entropies if e <= guess_entropy)
    return 100.0 * count / n


def luck_score(bucket_sizes: Sequence[int], actual_bucket_size: int) -> float:
    """Normalized bits-revealed luck (0-100) for `actual_bucket_size` within
    `bucket_sizes` (the non-empty bucket sizes of one guess, summing to the
    pool size N)."""
    sizes = list(bucket_sizes)
    n = sum(sizes)
    if n <= 0 or actual_bucket_size <= 0:
        return 0.0
    if actual_bucket_size not in sizes:
        raise ValueError(
            f"actual_bucket_size {actual_bucket_size} is not one of the guess's buckets"
        )

    k = actual_bucket_size
    k_max = max(sizes)
    info_actual = math.log2(n) - math.log2(k)
    info_max = math.log2(n)  # k == 1
    info_min = math.log2(n) - math.log2(k_max)
    if info_max == info_min:
        # Every non-empty bucket has size 1 (a perfect singleton split) or a
        # single bucket of size n; either way the outcome is deterministic.
        return 100.0 if k == 1 else 0.0
    return 100.0 * (info_actual - info_min) / (info_max - info_min)


def _bucket_sizes(analysis: "GuessAnalysis") -> dict[int, int]:
    """Per-score bucket sizes from a GuessAnalysis, preferring the raw
    `.buckets` mapping and falling back to compact `.bucket_counts`."""
    if analysis.buckets is not None:
        return {int(s): len(words) for s, words in analysis.buckets.items()}
    if analysis.bucket_counts is not None:
        return {int(s): int(c) for s, c in analysis.bucket_counts}
    raise ValueError("analysis carries no bucket data (buckets/bucket_counts are None)")


def skill(analysis: "GuessAnalysis", analyses: Sequence["GuessAnalysis"]) -> float:
    """Skill (0-100) of `analysis` among `analyses` (all guesses at the same
    state, e.g. from `SolverEngine.get_analyses()`)."""
    return skill_score(analysis.entropy, [a.entropy for a in analyses])


def luck(analysis: "GuessAnalysis", actual_score: int) -> float:
    """Luck (0-100) of the actual outcome `actual_score` (the packed base-3
    score, 0..242) for `analysis`'s guess."""
    sizes = _bucket_sizes(analysis)
    if actual_score not in sizes:
        raise ValueError(f"score {actual_score} is not a non-empty bucket of this guess")
    return luck_score(list(sizes.values()), sizes[actual_score])
