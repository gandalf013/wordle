"""Pluggable guess-ranking heuristics.

Pure: a Strategy does no scoring itself (analysis.analyze_all does that) and
holds no game state. Swapping heuristics -- or switching a heuristic between
weighted and uniform mode -- is a constructor argument, not a code change to
the game loop.
"""

import math
from typing import Protocol

from analysis import GuessAnalysis, bucket_counts_from_buckets


class Strategy(Protocol):
    requires_bucket_stats: bool = False

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

    requires_bucket_stats = False

    def __init__(self, tie_tol: float = 1e-9, weighted: bool = False):
        self.tie_tol = tie_tol
        self.weighted = weighted

    def _key(self, analysis: GuessAnalysis) -> float:
        if self.weighted and analysis.weighted_entropy is not None:
            return analysis.weighted_entropy
        return analysis.entropy

    def rank(
        self, analyses: list[GuessAnalysis]
    ) -> list[GuessAnalysis]:
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

    requires_bucket_stats = False

    def __init__(self, weighted: bool = False):
        self.weighted = weighted

    def _key(self, analysis: GuessAnalysis) -> float:
        if self.weighted and analysis.weighted_expected_size is not None:
            return analysis.weighted_expected_size
        return analysis.expected_size

    def rank(
        self, analyses: list[GuessAnalysis]
    ) -> list[GuessAnalysis]:
        return sorted(analyses, key=self._key)


class MinimaxStrategy:
    """Minimize the worst-case (largest) bucket -- classic Knuth-style
    solver. Deliberately has no weighted mode: "worst case" is an
    adversarial guarantee, and weighting it would contradict the point --
    an implausible-but-possible answer should still be guarded against.
    """

    requires_bucket_stats = False

    def rank(
        self, analyses: list[GuessAnalysis]
    ) -> list[GuessAnalysis]:
        return sorted(analyses, key=lambda a: a.worst_case_size)


class TwoPlyExpectimaxStrategy:
    """Two-ply expectimax strategy for Normal Mode Wordle.

    Evaluates how effectively candidate guesses split remaining candidate
    targets into buckets, and estimates the exact 2-turn resolution cost for
    each bucket.

    When `weighted=True`, weights buckets by probability mass rather than raw
    word count. Ties within `tie_tol` are broken toward candidate solutions.

    Ranking reads `bucket_counts`/`bucket_masses` off each GuessAnalysis (the
    compact bucket tallies populated by analyze_all's
    `include_bucket_stats=True`) -- the strategy does no scoring itself, and
    never needs the raw weights dict. SolverEngine enables the stats
    automatically via the `requires_bucket_stats` flag.
    """

    requires_bucket_stats = True

    def __init__(
        self,
        beam_width: int = 30,
        weighted: bool = False,
        tie_tol: float = 1e-9,
        large_n_anchor: tuple[int, float] = (3209, 3.6),
    ):
        self.beam_width = beam_width
        self.weighted = weighted
        self.tie_tol = tie_tol
        self.large_n_anchor = large_n_anchor

    def _estimate_bucket_cost(self, n: int) -> float:
        """Estimated expected number of *additional* guesses (beyond the
        guess that produced this bucket) needed to resolve a residual
        bucket of size `n`.

        n=0/1/2 are exact, analytically-derived values: an empty bucket
        needs nothing further; a lone survivor needs exactly one more
        guess; a 2-way tie needs one more guess half the time and two the
        other half (guess either candidate directly), so 0.5*1 + 0.5*2 =
        1.5. Larger buckets are interpolated *log-linearly* between the
        n=3 anchor (2.0 -- provably achievable: either a perfect
        1-guess/1/1/1 split exists, giving a guaranteed 2, or guessing a
        candidate directly gives the same 2.0 in expectation) and
        `large_n_anchor`, a (pool_size, avg_guesses) point measured from
        this repo's own EntropyStrategy performance
        (`benchmark_strategies.py`) -- resolving a bucket takes guesses
        proportional to the *information* (bits) needed to narrow it, not
        to its raw size, so the growth is ~log2(n), never linear.

        The previous version of this formula (`2.0 + 0.3 * (n - 3)`) was
        an unvalidated linear guess: it predicted needing ~963 *more*
        guesses to resolve a 3,209-word bucket, when real play resolves
        that in ~2.6. That miscalibration made this strategy effectively
        blind to bucket size beyond a handful of words, which is why it
        underperformed plain EntropyStrategy in benchmarks despite doing
        strictly more work.
        """
        if n <= 0:
            return 0.0
        if n == 1:
            return 1.0
        if n == 2:
            return 1.5
        lo_n, lo_cost = 3, 2.0
        hi_n, hi_cost = self.large_n_anchor
        if n >= hi_n:
            return hi_cost
        frac = (math.log2(n) - math.log2(lo_n)) / (math.log2(hi_n) - math.log2(lo_n))
        return lo_cost + (hi_cost - lo_cost) * frac

    @staticmethod
    def _counts_for(a: GuessAnalysis) -> tuple[tuple[int, int], ...] | None:
        """Per-guess bucket (score, count) pairs, falling back to deriving
        them from `.buckets` when `analyze` populated that instead of
        `bucket_counts`."""
        if a.bucket_counts is not None:
            return a.bucket_counts
        if a.buckets is not None:
            return bucket_counts_from_buckets(a.buckets)
        return None

    def _resolve_denominator(self, beam: list[GuessAnalysis], weighted_mode: bool) -> float:
        """denom is the pool total -- all targets (unweighted) or the total
        probability mass (weighted) -- which is the same for every guess,
        so it can be taken from any analysis that has the relevant stats."""
        for a in beam:
            counts = self._counts_for(a)
            if counts is None:
                continue
            if weighted_mode and a.bucket_masses is not None:
                denom = sum(m for _, m in a.bucket_masses)
                if denom > 0:
                    return denom
            if not weighted_mode:
                denom = sum(c for _, c in counts)
                if denom > 0:
                    return float(denom)
        return 1.0

    def rank(
        self, analyses: list[GuessAnalysis]
    ) -> list[GuessAnalysis]:
        if not analyses:
            return []

        base_strategy = EntropyStrategy(weighted=self.weighted, tie_tol=self.tie_tol)
        initial_ranked = base_strategy.rank(analyses)
        beam = initial_ranked[: self.beam_width]
        rest = initial_ranked[self.beam_width :]

        weighted_mode = self.weighted and any(
            a.bucket_masses is not None for a in beam
        )
        denom = self._resolve_denominator(beam, weighted_mode)

        scored_beam = []
        for a in beam:
            counts = self._counts_for(a)
            if counts is None or (weighted_mode and a.bucket_masses is None):
                # weighted_mode's denom is a probability-mass total; an entry
                # without bucket_masses of its own has no mass figures on the
                # same scale, so it can't be scored against that denom.
                scored_beam.append((float("inf"), a))
                continue

            # The win-score bucket (guess == target exactly) needs *zero*
            # additional guesses -- the game is already over. Without this,
            # every bucket-cost lookup (including _estimate_bucket_cost(1))
            # charged that branch a full extra guess, which systematically
            # made guessing an actual candidate look no better than wasting
            # a turn on a pure information-gathering probe.
            win_score = 3 ** len(a.guess) - 1

            if weighted_mode and a.bucket_masses is not None:
                cost = 1.0 + sum(
                    m * self._estimate_bucket_cost(c)
                    for (score, c), (_, m) in zip(counts, a.bucket_masses)
                    if score != win_score
                ) / denom
            else:
                cost = 1.0 + sum(
                    c * self._estimate_bucket_cost(c)
                    for score, c in counts
                    if score != win_score
                ) / denom
            scored_beam.append((cost, a))

        if not scored_beam or all(cost == float("inf") for cost, _ in scored_beam):
            return initial_ranked

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
