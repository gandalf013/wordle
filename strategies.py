"""Pluggable guess-ranking heuristics.

Pure: a Strategy does no scoring itself (analysis.analyze_all does that) and
holds no game state. Swapping heuristics -- or switching a heuristic between
weighted and uniform mode -- is a constructor argument, not a code change to
the game loop.
"""

import math
from typing import Protocol

import numpy as np

import fast_scoring
from analysis import GuessAnalysis


class Strategy(Protocol):
    def rank(
        self, analyses: list[GuessAnalysis], weights: dict[str, float] | None = None
    ) -> list[GuessAnalysis]:
        """Return `analyses` sorted best-first."""
        ...


def _move_to_front(analyses: list[GuessAnalysis], winner: GuessAnalysis) -> list[GuessAnalysis]:
    return [winner] + [a for a in analyses if a is not winner]


class EntropyStrategy:
    """Maximize information gain. If `weighted=True`, ranks by
    `weighted_entropy` (falling back to uniform `entropy` for any analysis
    where weights weren't supplied) instead of raw bucket-count entropy.
    Ties within `tie_tol` are broken toward a guess that is itself a
    possible solution -- and when weighted, toward the higher
    `solution_probability` among ties, not an arbitrary one.

    Matches Game.find_best_guess's selection when weighted=False for every
    case in TestFindBestGuess and the real-word-list round-1 golden value.
    One known divergence: Game.find_best_guess sorts via
    np.argsort(entropy)[::-1], which -- unlike Python's stable sorted(...,
    reverse=True) used here -- reverses the relative order of exactly-tied
    entries, not just tie groups. That only changes which guess is *tried
    first* when 3+ guesses share bit-identical entropy AND the very top one
    isn't itself a candidate solution; this is an accidental artifact of
    the reversal, not a rule worth preserving, so it isn't replicated here.
    """

    def __init__(self, tie_tol: float = 1e-9, weighted: bool = False):
        self.tie_tol = tie_tol
        self.weighted = weighted

    def _key(self, analysis: GuessAnalysis) -> float:
        if self.weighted and analysis.weighted_entropy is not None:
            return analysis.weighted_entropy
        return analysis.entropy

    def rank(
        self, analyses: list[GuessAnalysis], weights: dict[str, float] | None = None
    ) -> list[GuessAnalysis]:
        # `weights` is accepted (not used directly) to satisfy the Strategy
        # protocol; weighting is already baked into each analysis's
        # `weighted_entropy` by analyze_all, which is what `_key` reads.
        ordered = sorted(analyses, key=self._key, reverse=True)
        best = ordered[0]
        best_key = self._key(best)

        if not best.is_possible_solution:
            tied_candidates = []
            for candidate in ordered[1:]:
                if not math.isclose(
                    self._key(candidate), best_key, rel_tol=self.tie_tol, abs_tol=self.tie_tol
                ):
                    break
                if candidate.is_possible_solution:
                    tied_candidates.append(candidate)
                    if not self.weighted:
                        # Unweighted: first candidate found in the tied
                        # block wins, matching find_best_guess exactly.
                        break

            if tied_candidates:
                best = max(
                    tied_candidates, key=lambda a: a.solution_probability or 0.0
                )

        return _move_to_front(ordered, best)


class ExpectedPoolSizeStrategy:
    """Minimize the expected number of remaining candidates after this
    guess -- a 1-step-lookahead proxy for "minimize expected number of
    guesses". `weighted=True` uses `weighted_expected_size` (expected
    remaining *probability mass*, not raw count) so a guess that leaves 50
    near-impossible words as candidates isn't penalized the same as one
    that leaves 50 equally-plausible ones.
    """

    def __init__(self, weighted: bool = False):
        self.weighted = weighted

    def _key(self, analysis: GuessAnalysis) -> float:
        if self.weighted and analysis.weighted_expected_size is not None:
            return analysis.weighted_expected_size
        return analysis.expected_size

    def rank(
        self, analyses: list[GuessAnalysis], weights: dict[str, float] | None = None
    ) -> list[GuessAnalysis]:
        # See EntropyStrategy.rank: weighting already lives in
        # `weighted_expected_size`, which `_key` reads directly.
        return sorted(analyses, key=self._key)


class MinimaxStrategy:
    """Minimize the worst-case (largest) bucket -- classic Knuth-style
    solver. Deliberately has no weighted mode: "worst case" is an
    adversarial guarantee, and weighting it would contradict the point --
    an implausible-but-possible answer should still be guarded against.
    """

    def rank(
        self, analyses: list[GuessAnalysis], weights: dict[str, float] | None = None
    ) -> list[GuessAnalysis]:
        # `weights` is accepted only to satisfy the Strategy protocol -- see
        # the class docstring for why worst-case size is never weighted.
        return sorted(analyses, key=lambda a: a.worst_case_size)


class TwoPlyExpectimaxStrategy:
    """Two-ply expectimax strategy for Normal Mode Wordle.

    Evaluates how effectively candidate guesses split remaining candidate
    targets into buckets, and estimates the exact 2-turn resolution cost for
    each bucket.

    When `weighted=True`, weights buckets by probability mass rather than raw
    word count. Ties within `tie_tol` are broken toward candidate solutions.
    """

    def __init__(self, beam_width: int = 30, weighted: bool = False, tie_tol: float = 1e-9):
        self.beam_width = beam_width
        self.weighted = weighted
        self.tie_tol = tie_tol

    def _estimate_bucket_cost(self, n: int) -> float:
        if n <= 0:
            return 0.0
        if n == 1:
            return 1.0
        if n == 2:
            return 1.5
        return 2.0 + 0.3 * (n - 3)

    def rank(
        self, analyses: list[GuessAnalysis], weights: dict[str, float] | None = None
    ) -> list[GuessAnalysis]:
        if not analyses:
            return []

        base_strategy = EntropyStrategy(weighted=self.weighted, tie_tol=self.tie_tol)
        initial_ranked = base_strategy.rank(analyses, weights)
        beam = initial_ranked[: self.beam_width]
        rest = initial_ranked[self.beam_width :]

        target_pool = [a.guess for a in analyses if a.is_possible_solution]
        if not target_pool:
            target_pool = [a.guess for a in analyses]
        total_targets = len(target_pool)
        if total_targets == 0:
            return analyses

        beam_guesses = [a.guess for a in beam]
        matrix = fast_scoring.score_matrix(beam_guesses, target_pool)

        if self.weighted and weights is not None:
            target_weights = np.array([weights.get(w, 1.0) for w in target_pool], dtype=np.float64)
            total_mass = float(target_weights.sum())
        else:
            target_weights = None
            total_mass = float(total_targets)

        weighted_mode = self.weighted and target_weights is not None
        denom = total_mass if weighted_mode else total_targets

        counts_matrix, masses_matrix = fast_scoring.bincount_scores(matrix, weights=target_weights)

        scored_beam = []
        for i, a in enumerate(beam):
            counts = counts_matrix[i]
            masses = masses_matrix[i]
            active_mask = counts > 0
            active_counts = counts[active_mask]
            weighted_sum = masses[active_mask] if weighted_mode else active_counts
            b_costs = np.array([self._estimate_bucket_cost(int(c)) for c in active_counts])
            cost = 1.0 + float(np.sum(weighted_sum * b_costs)) / denom if denom else 1.0
            scored_beam.append((cost, a))

        scored_beam.sort(key=lambda item: item[0])
        best_cost, best = scored_beam[0]

        if not best.is_possible_solution:
            tied = []
            for cost, candidate in scored_beam[1:]:
                if not math.isclose(cost, best_cost, rel_tol=self.tie_tol, abs_tol=self.tie_tol):
                    break
                if candidate.is_possible_solution:
                    tied.append(candidate)
                    if not self.weighted:
                        break
            if tied:
                best = max(tied, key=lambda x: x.solution_probability or 0.0)

        ordered = [item[1] for item in scored_beam]
        return _move_to_front(ordered, best) + rest
