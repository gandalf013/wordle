"""Pluggable guess-ranking heuristics.

Pure: a Strategy does no scoring itself (analysis.analyze_all does that) and
holds no game state. Swapping heuristics -- or switching a heuristic between
weighted and uniform mode -- is a constructor argument, not a code change to
the game loop.
"""

import math
from typing import Protocol

from analysis import GuessAnalysis


class Strategy(Protocol):
    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]:
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

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]:
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

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]:
        return sorted(analyses, key=self._key)


class MinimaxStrategy:
    """Minimize the worst-case (largest) bucket -- classic Knuth-style
    solver. Deliberately has no weighted mode: "worst case" is an
    adversarial guarantee, and weighting it would contradict the point --
    an implausible-but-possible answer should still be guarded against.
    """

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]:
        return sorted(analyses, key=lambda a: a.worst_case_size)
