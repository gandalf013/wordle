"""Independent brute-force oracle for tiny word lists.

Deliberately dumb: no branch-and-bound, no lower bounds, no equivalence-
class dedup, no move ordering -- just "try every guess, recurse, take the
min." Slow, but its correctness is obvious by inspection, which is the
point: it shares no logic with wordle_solver.c, so agreement between the
two is real evidence the C solver's optimizations (bounds, pruning, dedup)
haven't changed the answer, not just self-consistency.

Only usable up to a handful of words (the recursion is exponential in the
worst case); see test_solver.py's ORACLE_CASES for sizes that stay fast.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring  # noqa: E402

EXACT_MATCH = 242


def _build_solver(guesses: tuple[str, ...]):
    """Returns solve(target_set) -> true min total guesses, free choice of
    guess (from `guesses`) at every step, memoized on the target subset.

    A guess that shares no information with the current subset (e.g. no
    overlapping letters with any remaining target) can partition it into a
    single bucket identical to the subset itself -- recursing into that
    would just re-ask the same question forever. `in_progress` detects that
    cycle (the same subset already being solved higher up this guess's own
    call stack) and reports it as infinitely costly, which is correct: a
    guess that never converges can never be optimal, and it must not be
    memoized as a real answer since a *different* call path may still
    resolve the same subset in finite cost via a different guess.
    """
    memo: dict[frozenset[str], int] = {}

    def solve(target_set: frozenset[str], in_progress: set[frozenset[str]]) -> float:
        n = len(target_set)
        if n <= 1:
            return n
        if target_set in memo:
            return memo[target_set]
        if target_set in in_progress:
            return math.inf

        in_progress.add(target_set)
        best = math.inf
        for g in guesses:
            buckets: dict[int, list[str]] = {}
            for t in target_set:
                buckets.setdefault(scoring.get_score(g, t), []).append(t)

            cost = n
            for s, bucket in buckets.items():
                if s == EXACT_MATCH:
                    continue
                cost += solve(frozenset(bucket), in_progress)
                # Sound short-circuit: cost only grows (or hits inf) as more
                # bucket costs are added in, so once it can't beat `best` it
                # never will.
                if cost >= best:
                    break
            best = min(best, cost)
        in_progress.discard(target_set)

        # `guesses` always covers `targets` (see module docstring's callers),
        # so guessing a still-live target itself -- a free exact-match hit
        # plus a strictly smaller remainder -- always terminates normally by
        # induction on n. `best` is therefore always finite here.
        memo[target_set] = best
        return best

    return lambda target_set: solve(target_set, set())


def exact_cost(targets: list[str], guesses: list[str]) -> int:
    """True minimum total guesses to resolve `targets` under optimal play,
    with free choice of guess (from `guesses`) at every node, including the
    first. Matches wordle_solver.c's `--all`/`--top` search.
    """
    solve = _build_solver(tuple(guesses))
    return solve(frozenset(targets))


def forced_opener_cost(targets: list[str], guesses: list[str], opener: str) -> int:
    """True minimum total guesses when the very first guess is forced to be
    `opener` (optimal play thereafter). Matches wordle_solver.c's --opener.
    """
    solve = _build_solver(tuple(guesses))
    target_set = frozenset(targets)
    n = len(target_set)

    buckets: dict[int, list[str]] = {}
    for t in target_set:
        buckets.setdefault(scoring.get_score(opener, t), []).append(t)

    cost = n
    for s, bucket in buckets.items():
        if s == EXACT_MATCH:
            continue
        cost += solve(frozenset(bucket))
    return cost
