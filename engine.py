"""SolverEngine: owns the candidate pool, weights, and guess history for
one game. No input()/print()/logging -- callers (cli.py, or a one-off
script) decide how to surface suggestions and results.
"""

from dataclasses import dataclass
from enum import Enum, auto

import analysis
import fast_scoring
import scoring
from analysis import GuessAnalysis
from strategies import Strategy


class RoundOutcome(Enum):
    CONTINUE = auto()
    SOLVED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class RoundResult:
    outcome: RoundOutcome
    candidates_remaining: int
    solution: str | None = None
    guesses_used: int = 0


class SolverEngine:
    """`weights` is carried alongside `candidates` (keyed by word, not
    index, since the candidate pool's *contents* change every round but a
    word's weight doesn't) and passed through to analysis.analyze/
    analyze_all on every call -- so switching self.strategy between a
    weighted and unweighted variant changes ranking behavior without
    touching engine state at all.
    """

    def __init__(
        self,
        guess_list: list[str],
        target_list: list[str],
        strategy: Strategy,
        weights: dict[str, float] | None = None,
        initial_guess: str | None = None,
        show_progress: bool | None = None,
    ):
        self.guess_list = list(guess_list)
        self.target_list = list(target_list)
        self.candidates = list(target_list)
        self.strategy = strategy
        self.weights = weights
        self.initial_guess = initial_guess
        self.show_progress = show_progress
        self.n = len(self.guess_list[0])
        self.history: list[tuple[str, int]] = []

        # The first-round suggestion (whether from initial_guess or
        # self.strategy) is expensive to compute against the full word
        # list, and doesn't change across reset() cycles -- caching it here
        # is the direct replacement for Game's self.best_initial_guess.
        self._cached_initial_suggestion: GuessAnalysis | None = None
        self._cached_analyses: list[GuessAnalysis] | None = None
        self._cached_analyses_by_guess: dict[str, GuessAnalysis] | None = None

    def get_analyses(self) -> list[GuessAnalysis]:
        """Calculates or returns cached analyses for the current candidate pool."""
        if self._cached_analyses is None:
            use_cache = not self.history
            self._cached_analyses = analysis.analyze_all(
                self.guess_list,
                self.candidates,
                weights=self.weights,
                use_cache=use_cache,
                include_bucket_stats=self.strategy.requires_bucket_stats,
                show_progress=self.show_progress,
            )
            self._cached_analyses_by_guess = {
                a.guess: a for a in self._cached_analyses
            }
        return self._cached_analyses

    def get_ranked_analyses(self) -> list[GuessAnalysis]:
        """Return analyses for the current candidate pool ranked by strategy."""
        return self.strategy.rank(self.get_analyses())

    def suggest(self) -> GuessAnalysis:
        """Best guess for the current candidate pool, per self.strategy,
        using self.weights."""
        if not self.history:
            if self.initial_guess is not None:
                return analysis.analyze(
                    self.initial_guess,
                    self.candidates,
                    weights=self.weights,
                    use_cache=True,
                    show_progress=self.show_progress,
                )
            if self._cached_initial_suggestion is None:
                self._cached_initial_suggestion = self.get_ranked_analyses()[0]
            return self._cached_initial_suggestion

        return self.get_ranked_analyses()[0]

    def analyze(self, word: str, include_buckets: bool = True) -> GuessAnalysis:
        """Analyze `word` against the current pool (with weights) without
        committing to it. Does not touch self.candidates or self.history.

        When `include_buckets=False`, a full-round analysis already computed
        for the current pool is reused if `word` is in it, so the REPL's
        `analyze <word>` peek doesn't re-score the pool. Buckets (and any
        bucket stats) are never carried on those cached analyses, so the
        `buckets` command (include_buckets=True) always re-analyzes."""
        if self._cached_analyses_by_guess is not None and not include_buckets:
            cached = self._cached_analyses_by_guess.get(word)
            if cached is not None:
                return cached
        return analysis.analyze(
            word,
            self.candidates,
            weights=self.weights,
            use_cache=not self.history,
            show_progress=self.show_progress if include_buckets else False,
        )

    def apply_score(self, guess: str, score: int) -> RoundResult:
        """Commit to `guess` scoring `score`: narrows candidates, appends
        history. self.weights is never modified -- narrowing the candidate
        pool doesn't change any surviving word's weight, just which words
        survive.

        `history` records only moves actually played; when the pool collapses
        to a single candidate that wasn't itself the last guess, the final
        implied "guess it" move is *not* fabricated into history -- it's
        accounted for by RoundResult.guesses_used, the single source of truth
        for total guesses (callers like cli and the benchmark both read it)."""
        self._cached_analyses = None
        self._cached_analyses_by_guess = None
        self.history.append((guess, score))

        if len(self.candidates) > 50:
            scores = fast_scoring.score_matrix([guess], self.candidates)[0]
            new_candidates = [word for word, s in zip(self.candidates, scores) if s == score]
        else:
            new_candidates = [
                word for word in self.candidates if scoring.get_score(guess, word) == score
            ]
        if not new_candidates:
            return RoundResult(outcome=RoundOutcome.ERROR, candidates_remaining=0)

        self.candidates = new_candidates
        if len(new_candidates) > 1:
            return RoundResult(
                outcome=RoundOutcome.CONTINUE, candidates_remaining=len(new_candidates)
            )

        solution = new_candidates[0]
        return RoundResult(
            outcome=RoundOutcome.SOLVED,
            candidates_remaining=1,
            solution=solution,
            guesses_used=len(self.history) if guess == solution else len(self.history) + 1,
        )

    def reset(self) -> None:
        """Start a new round against the original target list. The cached
        initial suggestion survives a reset -- it depends only on
        guess_list/target_list/weights/strategy, none of which change."""
        self.candidates = list(self.target_list)
        self.history = []
        self._cached_analyses = None
        self._cached_analyses_by_guess = None
