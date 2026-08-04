# Wordle Solver Refactor Plan

Status as of 2026-08-04. This is a proposal — nothing in this document is
implemented yet, except where explicitly marked "already implemented"
below. It exists so a future session (with or without this conversation's
history) can pick up the plan without re-deriving it.

## Already implemented (as of this writing)

- `tests/test_wordle.py` — a behavior-pinning regression suite for the
  current monolithic `Game` class, `parse_file`, and scoring semantics.
  42-55 tests, runs in a few seconds. This is the safety net the refactor
  below must keep green throughout.
- `fast_scoring.py` — a batched NumPy scorer (`score_matrix`) that computes
  the full (guess × target) packed-score matrix, plus `cached_score_matrix`,
  which persists it to `.wordle_cache/` (gitignored) keyed by a content
  hash of the exact word lists. `Game.get_all_censuses`/`score_guess`
  already delegate to this. This is the performance foundation the
  refactor's `analysis.py` (below) should be built on top of, not
  duplicate.
- `words.weighted.txt` — 3,209 words sourced from WordleTools'
  weighted-bottles table, one `word <weight>` pair per line, no separate
  target/extra split (the list serves as both guesses and targets).
  `parse_file` currently parses the weight column but **discards it** —
  nothing downstream (`Game`, `run()`) sees it yet. Wiring weights through
  to the strategy layer is a first-class part of this plan, not an
  afterthought.

## Features driving this refactor

Things the user wants to add that the current single-`Game`-class design
makes hard:

1. Pluggable guess-selection heuristics (entropy maximization is the only
   one today) — e.g. minimize expected remaining pool size, minimax
   worst-case bucket.
2. **Weighted frequencies as a first-class input to those heuristics** —
   `words.weighted.txt` gives a relative likelihood per word; strategies
   should be able to rank guesses by weighted entropy / weighted expected
   pool size, not just raw counts over the candidate pool.
3. "Show me the entropy of `<word>` without committing to that guess."
4. A nicer display of word/entropy pairs for the top N guesses.
5. Displaying the "buckets" (score-pattern → matching words) a candidate
   guess splits the remaining pool into.

## What's wrong with the current structure

Everything lives in one `Game` class (`wordle.py`) that mixes:

- **scoring math** (`get_score`, encode/decode) — pure, stateless
- **strategy** (`find_best_guess`) — hardcoded to "maximize entropy,
  tie-break toward possible solutions," no weight awareness
- **state/rules** (`target_lists`, `reset`, round narrowing)
- **I/O** (`input()`, `logging.info()`, `sys.stdout.write()`) — interleaved
  directly into `play_one_round`/`get_guess_score`

The I/O coupling is the real blocker: there's no way to ask "what would the
entropy of `crane` be right now" without going through the interactive
prompt machinery. And nothing has a slot for weights to flow into a
ranking decision — `find_best_guess` only ever sees `Counter`-style bucket
*counts*, never word-level likelihoods.

## Proposed architecture

```
wordle/          (or flat modules next to wordle.py -- see "Open question" below)
  scoring.py      # Score enum, get_score, encode/decode -- pure, unchanged in spirit
  fast_scoring.py # already implemented -- vectorized matrix + cache, unchanged
  wordlists.py    # parse_file -- now returns weights too
  analysis.py     # census/entropy/bucket stats, weight-aware, built on fast_scoring
  strategies.py   # pluggable ranking, weight-aware
  engine.py       # SolverEngine: state machine, carries weights, zero I/O
  display.py      # all formatting, including weighted columns
  cli.py          # argparse + REPL command loop, the only place with input()/print()
wordle.py         # thin shim: from wordle.cli import main
```

### `wordlists.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class WordList:
    """Parsed word list. `weights` covers every word in target+extra;
    words with no explicit weight column default to 1.0 (uniform), so
    words.wordle.txt (no weights) and words.weighted.txt (weighted) are
    interchangeable inputs downstream -- callers never need to branch on
    which kind of file they got.
    """
    target: list[str]
    extra: list[str]
    word_length: int
    weights: dict[str, float]


def parse_file(fp) -> WordList:
    """Parse target words, a blank line, then extra guess-only words.
    Each line is 'word' or 'word <weight>'.
    """
    ...
```

This changes `parse_file`'s return type from today's 3-tuple to a
dataclass — a deliberate breaking change during the refactor (today's
callers, `run()` and the tests, get updated in the same step).

### `analysis.py`

```python
from dataclasses import dataclass
from typing import Sequence

import fast_scoring


@dataclass(frozen=True)
class GuessAnalysis:
    """Everything derived from scoring one candidate guess against a target
    pool. Computed once per (guess, pool[, weights]) triple; every strategy
    and every display view reads off this instead of each recomputing
    scores independently.

    The weighted_* fields are None when no weights were supplied to
    analyze() -- callers must not assume they're populated.
    """
    guess: str
    buckets: dict[int, list[str]]         # packed score -> matching words
    entropy: float                        # bits, uniform over bucket counts
    worst_case_size: int                  # size of the largest bucket
    expected_size: float                  # sum(p_i * bucket_size_i), uniform
    is_possible_solution: bool

    weighted_entropy: float | None = None
    # bits, computed over each bucket's total weight mass (normalized to a
    # probability distribution) instead of raw word counts. This is what
    # stops a guess that only splits low-probability chaff finely from
    # outranking one that cleanly separates the likely actual answers.
    weighted_expected_size: float | None = None
    # weight-mass-weighted expected remaining pool size after this guess.
    solution_probability: float | None = None
    # this guess's own relative weight / sum(weights over the pool) --
    # "how likely is THIS WORD to be the actual answer right now."


def analyze(
    guess: str,
    target_pool: Sequence[str],
    weights: dict[str, float] | None = None,
) -> GuessAnalysis:
    """Score `guess` against every word in `target_pool` (via
    fast_scoring.score_matrix, not a fresh Python loop) and summarize the
    split, including weighted variants if `weights` is given.

    Single entry point used by:
      - strategies, to rank candidate guesses
      - the `analyze <word>` REPL command (peek without committing)
      - the `buckets <word>` REPL command (renders `.buckets` directly)
    """
    ...


def analyze_all(
    guess_list: Sequence[str],
    target_pool: Sequence[str],
    weights: dict[str, float] | None = None,
) -> list[GuessAnalysis]:
    """analyze() for every candidate guess, backed by a single
    fast_scoring.cached_score_matrix/score_matrix call rather than one
    matrix build per guess -- this is the direct replacement for today's
    Game.get_all_censuses, now weight-aware."""
    ...
```

Why weights live here and not in `fast_scoring.py`: scoring (which letters
are green/yellow/gray) never depends on word likelihood, only on the
letters themselves. Keeping weights out of the cached score matrix means
the (expensive, rarely-changing) matrix cache stays valid even if
WordleBot-style weights get retuned later — only this cheap aggregation
step would need to rerun, not a full matrix rebuild.

### `strategies.py`

```python
from typing import Protocol


class Strategy(Protocol):
    """Ranks candidate guesses from best to worst, given their analyses.
    Pure: does no scoring itself (analysis.analyze_all does that) and
    holds no game state. Swapping heuristics -- or switching a heuristic
    between weighted and uniform mode -- is a constructor argument, not a
    code change to the game loop.
    """

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]:
        """Return `analyses` sorted best-first."""
        ...


class EntropyStrategy:
    """Maximize information gain. If `weighted=True`, ranks by
    `weighted_entropy` (falling back to uniform `entropy` for any analysis
    where weights weren't supplied) instead of raw bucket-count entropy.
    Ties within `tie_tol` are broken toward a guess that is itself a
    possible solution -- and when weighted, toward the higher
    `solution_probability` among ties, not an arbitrary one.
    """

    def __init__(self, tie_tol: float = 1e-9, weighted: bool = False): ...

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]: ...


class ExpectedPoolSizeStrategy:
    """Minimize the expected number of remaining candidates after this
    guess -- a 1-step-lookahead proxy for "minimize expected number of
    guesses". `weighted=True` uses `weighted_expected_size` (expected
    remaining *probability mass*, not raw count) so a guess that leaves 50
    near-impossible words as candidates isn't penalized the same as one
    that leaves 50 equally-plausible ones.
    """

    def __init__(self, weighted: bool = False): ...

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]: ...


class MinimaxStrategy:
    """Minimize the worst-case (largest) bucket -- classic Knuth-style
    solver. Deliberately has no weighted mode: "worst case" is an
    adversarial guarantee, and weighting it would contradict the point --
    an implausible-but-possible answer should still be guarded against.
    """

    def rank(self, analyses: list[GuessAnalysis]) -> list[GuessAnalysis]: ...
```

### `engine.py`

```python
from dataclasses import dataclass
from enum import Enum, auto


class RoundOutcome(Enum):
    CONTINUE = auto()
    SOLVED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class RoundResult:
    outcome: RoundOutcome
    candidates_remaining: int
    solution: str | None = None


class SolverEngine:
    """Owns the candidate pool, weights, and guess history for one game.
    No input()/print()/logging -- callers (cli.py, or a one-off script)
    decide how to surface suggestions and results.

    `weights` is carried alongside `candidates` (keyed by word, not index,
    since the candidate pool's *contents* change every round but a word's
    weight doesn't) and passed through to analysis.analyze/analyze_all on
    every call -- so switching self.strategy between a weighted and
    unweighted variant changes ranking behavior without touching engine
    state at all.
    """

    guess_list: list[str]
    candidates: list[str]
    weights: dict[str, float]
    history: list[tuple[str, int]]
    strategy: Strategy

    def __init__(
        self,
        guess_list: list[str],
        target_list: list[str],
        strategy: Strategy,
        weights: dict[str, float] | None = None,
        initial_guess: str | None = None,
    ): ...

    def suggest(self) -> GuessAnalysis:
        """Best guess for the current candidate pool, per self.strategy,
        using self.weights."""
        ...

    def analyze(self, word: str) -> GuessAnalysis:
        """Analyze `word` against the current pool (with weights) without
        committing to it. Does not touch self.candidates or self.history."""
        ...

    def apply_score(self, guess: str, score: int) -> RoundResult:
        """Commit to `guess` scoring `score`: narrows candidates, appends
        history. self.weights is never modified -- narrowing the candidate
        pool doesn't change any surviving word's weight, just which words
        survive."""
        ...

    def reset(self) -> None:
        """Start a new round against the original target list."""
        ...
```

### `display.py`

```python
def format_score(score: int, n: int) -> str:
    """Emoji rendering of a packed score, e.g. '⬛🟨🟩⬛🟩'."""
    ...


def format_top_guesses(
    analyses: list[GuessAnalysis], top_n: int = 10, weighted: bool = False
) -> str:
    """Table of the top-n analyses. When weighted=True and the analyses
    carry weighted fields, adds P(answer) and weighted-entropy columns
    alongside the uniform ones, so you can see both views at once rather
    than losing the raw-count numbers.
    """
    ...


def format_buckets(analysis: GuessAnalysis, limit: int | None = None) -> str:
    """Score pattern -> bucket size (+ sample words), largest buckets
    first. If the analysis carries weights, sort/annotate by bucket weight
    mass instead of raw count."""
    ...


def format_history(history: list[tuple[str, int]], n: int) -> str:
    """The 'guess EMOJI_ROW' lines shown when a game is solved."""
    ...
```

### `cli.py`

```python
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Score_:          # renamed to avoid clashing with scoring.Score
    value: int

@dataclass(frozen=True)
class OverrideGuess:
    word: str

@dataclass(frozen=True)
class Analyze:
    word: str

@dataclass(frozen=True)
class Buckets:
    word: str | None   # None -> use the current suggestion

@dataclass(frozen=True)
class Top:
    n: int

@dataclass(frozen=True)
class Restart: ...

@dataclass(frozen=True)
class Quit: ...

Command = Union[Score_, OverrideGuess, Analyze, Buckets, Top, Restart, Quit]


def parse_command(raw: str, n: int) -> Command:
    """Parse one line of REPL input into a Command.

    Grammar:
      '<n digits of 0/1/2>'        -> Score_(value)
      '!<word>'                    -> OverrideGuess(word)
      '?<word>' | 'analyze <word>' -> Analyze(word)
      'buckets [word]'             -> Buckets(word or None)
      'top [N]'                    -> Top(n=N or default 10)
      'r' | 'restart'              -> Restart()
      'q' | 'quit'                 -> Quit()

    Replaces the old `.islower()` shape-sniffing (which broke down once
    weights made "is this a score, a word, or something else" genuinely
    ambiguous) with explicit prefixes.
    """
    ...


def run_interactive(engine: SolverEngine, automatic: bool, solution: str | None) -> None:
    """The REPL: print suggestions, read+dispatch commands, print results.
    Analyze/Buckets/Top loop back without touching engine state;
    Score_/OverrideGuess drive engine.apply_score.
    """
    ...


def main(argv: list[str] | None = None) -> None:
    """Entry point: argparse (including a --weighted / --strategy flag
    selecting which Strategy instance to build) + dispatch to
    run_interactive."""
    ...
```

## Migration order (incremental, each step independently shippable)

1. **Done.** `wordlists.py`: extracted `parse_file` as `WordList`, weights
   included. `wordle.py`'s `run()` and `tests/test_wordle.py` updated to
   the new return shape; `wordle.py` no longer defines its own
   `parse_file`, it imports from `wordlists`. Weights are now actually
   parsed into a `word -> float` dict (defaulting to 1.0 when no weight
   column is present) instead of being discarded. Full test suite (56
   tests, including the slow real-word-list golden-value test) passes.
2. **Done.** `analysis.py`: `GuessAnalysis` + `analyze`/`analyze_all`,
   implemented on top of the *existing*
   `fast_scoring.score_matrix`/`cached_score_matrix` (no scoring
   reimplemented). Uniform fields only so far (`weighted_*` fields exist on
   `GuessAnalysis` but always come back `None` -- wiring them up is step 3).
   `analyze_all` takes a `use_cache` flag mirroring `Game.get_all_censuses`'s
   `self.round == 0` cache guard, left for the caller to set rather than
   inferred, since this module carries no round state.
   `tests/test_analysis.py` cross-checks `analyze`/`analyze_all` against
   `Game.get_score`/`get_all_censuses`/`get_all_entropy`, including a
   real-word-list parity test that reproduces the round-1 golden value
   (`tarse`, entropy ≈5.948974509955522). 70 tests total pass (68 fast, 2
   slow).
3. **Done.** `weights` parameter and `weighted_*` fields wired through
   `analyze`/`analyze_all`. Missing entries in `weights` default to 1.0
   (matching `WordList`'s own default), and `solution_probability` is
   forced to 0.0 for a guess that isn't itself in `target_pool`, regardless
   of what the weights dict says about it, since it can't be the answer.
   `tests/test_analysis.py::TestWeightedAnalyze` has a hand-constructed
   6-word pool (two high-weight "plausible" targets + four near-zero-weight
   "chaff" targets) with two guesses, `dc` and `bb`, where uniform entropy
   and weighted entropy pick opposite winners -- `dc` finely splits the
   chaff (higher raw entropy) but lumps the two plausible answers together
   (near-zero weighted entropy); `bb` does the reverse. This is the
   regression test that would catch weighting being silently a no-op. 77
   tests total pass (75 fast, 2 slow).
4. **Done.** `strategies.py`: `Strategy` protocol, `EntropyStrategy`,
   `ExpectedPoolSizeStrategy`, `MinimaxStrategy`. `EntropyStrategy(
   weighted=False)` matches `Game.find_best_guess` on every case in
   `TestFindBestGuess` and the real-word-list round-1 golden value
   (`tarse`, entropy ≈5.948974509955522). One deliberate, documented
   non-parity: `Game.find_best_guess` sorts via `np.argsort(entropy)[::-1]`,
   which (unlike Python's stable `sorted(..., reverse=True)` used here)
   reverses the relative order of *exactly*-tied entries, not just tie
   groups; this only changes which guess wins a 3+-way bit-identical
   entropy tie where the top-sorted entry isn't itself a candidate --
   verified to actually diverge with a constructed repro
   (`Game(["az","aa","bb"], ["aa","bb"])` picks `"bb"`;
   `EntropyStrategy().rank(...)` picks `"aa"`) -- and is judged an
   accidental artifact of the reversal, not a rule worth replicating.
   `EntropyStrategy(weighted=True)`'s tie-break-toward-highest-
   `solution_probability` and `ExpectedPoolSizeStrategy`'s weighted-vs-
   uniform disagreement are tested against hand-constructed
   `GuessAnalysis` instances (built directly, not derived from real word
   scoring) rather than searched-for real words, since exact float ties
   are impractical to hit reliably via real bucket scoring — confirmed by
   two failed attempts to hand-derive such examples from real 2-letter
   word buckets before switching approach. 86 tests total pass (83 fast,
   3 slow).
5. **Done.** `Game` retired; `scoring.py` (pure `get_score`/`get_score_num`/
   `get_score_list`/`get_score_str`, extracted from `Game`'s scalar
   methods), `engine.py` (`SolverEngine`, `RoundOutcome`, `RoundResult` —
   zero I/O), and `cli.py` (the interactive loop: `resolve_score`,
   `play_one_round`, `run_interactive`, `run`, `main`) now do what `Game`
   used to. `wordle.py` is the thin shim from the proposed architecture
   (`from cli import main`). `main()` wires `SolverEngine(weights=
   word_list.weights)` by default, using `EntropyStrategy()` (unweighted,
   matching prior behavior) since no `--strategy`/`--weighted` flag exists
   yet (step 7).

   `SolverEngine.suggest()` reproduces `Game`'s `best_initial_guess`
   caching and `initial_guess` override exactly (both survive `reset()`).
   Test files were split along the same lines as the modules:
   `test_scoring.py`, `test_fast_scoring.py`, `test_wordlists.py` (all
   ported from the old `test_wordle.py`, updated to call functions instead
   of `Game` methods), `test_engine.py` (new, ports
   `TestGetGuessScore`/`TestPlayOneRound`'s non-I/O assertions), `test_cli.py`
   (new, ports the same tests' I/O-mocking assertions). `test_analysis.py`
   and `test_strategies.py` no longer use `Game` as a live comparison
   oracle (it's gone) — their parity checks were changed to an independent
   numpy/scipy census/entropy reimplementation (`test_analysis.py`) or
   hardcoded assertions carrying forward values already cross-verified
   against `Game` in earlier steps (`test_strategies.py`).

   One feature was deliberately dropped, not ported: `-n`/
   `--num-top-guesses`. Reproducing it would mean duplicating
   `SolverEngine.suggest()`'s internal branching just to peek at the ranked
   list one level down; it's superseded by a proper `top` REPL command in
   step 7 instead of carrying forward as a CLI flag. One non-obvious,
   verified-not-a-bug behavior difference: round-2 in a manual end-to-end
   run picked `abaci` over `Game`'s `zymic` after guessing `tarse` against
   `crate` — both have bit-identical entropy (`log2(3)`, neither is a
   candidate), so this is the same accidental numpy-tie-reversal artifact
   documented in step 4, now observed in the wild rather than only in a
   constructed repro. 92 tests total pass (90 fast, 2 slow); CLI smoke-tested
   end-to-end (automatic+solution, interactive score entry, restart).
6. **Done.** `display.py`: `format_score`, `format_history`,
   `format_top_guesses`, `format_buckets` — all pure str-in/str-out. As
   part of this, `SQUARES`/`get_score_str` moved out of `scoring.py` (where
   step 5 had put them as an interim choice) into `display.py`'s
   `format_score`, matching the plan's original module boundary: `scoring.py`
   only knows packed ints and their decoded `Score` list, never emoji
   rendering. `format_score` also drops `get_score_str`'s dual-mode
   int-or-list input in favor of a single int-only signature (per the
   plan's `format_score(score: int, n: int)`), since every real caller
   already had a packed int in hand. `cli.py` now calls
   `display.format_score`/`format_history` instead of building score
   strings itself. `format_top_guesses` and `format_buckets` aren't wired
   into `cli.py` yet — nothing calls them until step 7's `analyze`/
   `buckets`/`top` REPL commands exist, but they're implemented and tested
   now (mirroring how step 4 built all three `Strategy` implementations
   before `engine.py` used more than one). `format_buckets` takes an
   optional `weights` param not in the plan's literal signature: sorting
   by bucket weight *mass* needs per-word weights, which aren't stored on
   `GuessAnalysis` (only the already-aggregated `weighted_*` summary
   fields are) — a deliberate, documented deviation from the proposal, not
   an oversight. New `tests/test_display.py`; the `get_score_str` tests
   moved out of `test_scoring.py` into it. 102 tests total pass (100 fast,
   2 slow); CLI smoke-tested end-to-end, output byte-identical to before
   the move.
7. **Done.** `parse_command`/`Command` (`Score_`, `OverrideGuess`, `Analyze`,
   `Buckets`, `Top`, `Restart`, `Quit`) replace the old `.islower()`
   shape-sniffing from step 5 with explicit-prefix parsing, exactly as
   specced. `play_one_round` became a single unified per-round loop
   instead of the old two-stage "guess-or-override prompt, then a separate
   score prompt": it tracks a `current_guess` (updated by `OverrideGuess`),
   with `Analyze`/`Buckets`/`Top` looping back without touching engine
   state and `Score_`/`Restart`/`Quit` ending the round. `main()` gained
   `--strategy {entropy,expected-pool-size,minimax}` and `--weighted`,
   wired through a small `build_strategy()` that logs a warning and drops
   `--weighted` for `minimax` (which has no weighted mode by design, see
   step 4) rather than erroring.

   Two deliberate, documented departures from a literal one-to-one port of
   the old two-prompt flow, both consequences of unifying to one prompt
   per round:
   - When `solution` is known, an **empty** line now accepts the current
     guess and lets the solution resolve its score, instead of requiring a
     throwaway `Score_` value the solution was going to override anyway
     (it still did override manually-typed digits when given, preserving
     that priority exactly — see `test_solution_overrides_a_manually_typed_score`).
   - `-a`/`--automatic` now only skips prompting outright when `solution`
     is also known (the original had the same actual behavior — `-a`
     without `-s` always fell through to a manual score prompt — this just
     makes that explicit rather than incidental, since the old
     two-separate-prompts structure that produced it no longer exists to
     produce it by accident).

   `-n`/`--num-top-guesses` (dropped in step 5) is now properly superseded:
   `top [N]` gives the same information on demand instead of forced onto
   every round. Manual end-to-end smoke tests against the real word list
   covered all four new/changed pieces: `?word`/`buckets`/`top N` peeking
   mid-round without committing, `!word` overriding the suggestion,
   `--strategy expected-pool-size --weighted` picking a different first
   guess (`roate`) than the default entropy strategy (`tarse`), and the
   `-a -s` fast-path still working unchanged. 125 tests total pass (123
   fast, 2 slow) — this was the last step in the migration order; the
   refactor described in this document is now complete.

## Explicitly out of scope

- True lookahead-based "expected number of steps" (recursive
  minimax/expectimax over future rounds) — `ExpectedPoolSizeStrategy` is a
  1-step proxy, not this. Worth its own design later if it turns out to
  matter in practice.
- A WordleBot-style "skill vs. luck" post-game score. The weighted
  `GuessAnalysis`/`solution_probability` fields above are the building
  blocks it would need (skill ≈ how close each guess was to the
  weighted-optimal choice at that step; luck ≈ how improbable the actual
  answer was under the weights), but assembling that into an actual report
  is a separate feature on top of this refactor, not part of it.
- Performance work beyond what `fast_scoring.py` already does.

## Open question (deferred, not blocking)

Flat modules next to `wordle.py` (as sketched above) vs. a `wordle/`
package directory. Flat is lower-ceremony and matches this repo's current
style; a package only earns its keep if this becomes pip-installable.
Default to flat unless that changes.
