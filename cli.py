"""cli.py: the interactive REPL and CLI entry point built on SolverEngine.
The only module in the wordle solver with input()/print()/logging calls --
scoring, analysis, strategy selection, and display formatting are all pure
by this point, so this module's only job is deciding how to surface them
interactively (calling display.py's formatters, never building formatted
strings itself).

The REPL grammar (parse_command/Command) replaces the old `.islower()`
shape-sniffing from before this step: back when the only two things a line
of input could mean were "a score" or "a guess override", checking
`len(s) == n and s.islower()` was enough to tell them apart. Once `analyze`/
`buckets`/`top` needed their own syntax too, shape-sniffing stopped scaling,
so every command now has an explicit, unambiguous prefix/keyword instead.

One feature from before this step was deliberately never ported: the old
-n/--num-top-guesses flag, which logged the top N guesses by entropy on
every round unconditionally. The `top` REPL command below supersedes it --
available on demand instead of forced on every round.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import analysis
import display
import scoring
from engine import RoundOutcome, SolverEngine
from strategies import (
    EntropyStrategy,
    ExpectedPoolSizeStrategy,
    MinimaxStrategy,
    Strategy,
    TwoPlyExpectimaxStrategy,
)
from wordlists import parse_file


class LoopState(Enum):
    QUIT = auto()
    RESTART = auto()
    CONTINUE = auto()
    ERROR = auto()
    SOLVED = auto()


@dataclass(frozen=True)
class ShowScore:  # renamed to avoid clashing with scoring.Score
    value: int


@dataclass(frozen=True)
class OverrideGuess:
    word: str


@dataclass(frozen=True)
class Analyze:
    word: str


@dataclass(frozen=True)
class Buckets:
    word: str | None  # None -> use the current guess


@dataclass(frozen=True)
class Top:
    n: int


@dataclass(frozen=True)
class Restart:
    pass


@dataclass(frozen=True)
class Quit:
    pass


@dataclass(frozen=True)
class Help:
    pass


Command = Union[ShowScore, OverrideGuess, Analyze, Buckets, Top, Restart, Quit, Help]

_DEFAULT_TOP_N = 10

_HELP_TEXT = """\
Commands:
  <n digits of 0/1/2> -> record this score for the current guess
  !<word> -> override the current guess with <word>
  ?<word> | analyze <word> -> analyze <word> as a candidate guess
  buckets [word] -> show score-bucket breakdown for [word] (or current guess)
  top [N] -> show the top N guesses by the active strategy (default 10)
  r | restart -> restart the round
  q | quit -> quit
  ? -> show this help"""


def parse_command(raw: str, n: int) -> Command | None:
    """Parse one line of REPL input into a Command, or None if it doesn't
    match any known form.

    Grammar:
      '<n digits of 0/1/2>'        -> ShowScore(value)
      '!<word>'                    -> OverrideGuess(word)
      '?<word>' | 'analyze <word>' -> Analyze(word)
      'buckets [word]'             -> Buckets(word or None)
      'top [N]'                    -> Top(n=N or default 10)
      'r' | 'restart'              -> Restart()
      'q' | 'quit'                 -> Quit()
      '?'                          -> Help()

    A word after `!`/`?`/`analyze` that isn't exactly `n` letters is
    rejected (returns None) rather than accepted and left to fail later
    inside scoring -- the same length check the old override-detection
    logic did before this grammar existed.
    """
    s = raw.strip()
    if not s:
        return None

    parts = s.split(maxsplit=1)
    head = parts[0].lower()
    rest = parts[1].strip().lower() if len(parts) > 1 else None

    if head in ("r", "restart"):
        return Restart()
    if head in ("q", "quit"):
        return Quit()

    if head == "analyze" and rest and len(rest) == n:
        return Analyze(rest)

    if head == "buckets":
        if rest is not None and len(rest) != n:
            return None
        return Buckets(rest)

    if head == "top":
        if rest is None:
            return Top(_DEFAULT_TOP_N)
        try:
            return Top(int(rest))
        except ValueError:
            return None

    if s.startswith("!"):
        word = s[1:].strip().lower()
        return OverrideGuess(word) if word and len(word) == n else None

    if s == "?":
        return Help()

    if s.startswith("?"):
        word = s[1:].strip().lower()
        return Analyze(word) if word and len(word) == n else None

    if len(s) == n and all(c in "012" for c in s):
        return ShowScore(int(s, base=3))

    return None


def _commit(engine, guess, score, threshold_display):
    sys.stdout.write(f"{display.format_score(score, engine.n)}\n")

    result = engine.apply_score(guess, score)
    if result.outcome == RoundOutcome.ERROR:
        logging.error("No guess matches the score!")
        return LoopState.ERROR

    if result.outcome == RoundOutcome.SOLVED:
        logging.info(f"SOLVED: {result.solution} in {len(engine.history)} guesses")
        sys.stdout.write(display.format_history(engine.history, engine.n) + "\n")
        return LoopState.SOLVED

    logging.info(f"{result.candidates_remaining} words match the pattern")
    if result.candidates_remaining <= threshold_display:
        logging.info(f"Matching words: {sorted(engine.candidates)}")
    logging.debug(f"Words:\n{engine.candidates}")

    return LoopState.CONTINUE


def play_one_round(engine, automatic, solution, threshold_display=3):
    suggestion = engine.suggest()
    current_guess = suggestion.guess
    logging.info(f"Best guess {current_guess} entropy {suggestion.entropy}")

    if automatic and solution is not None:
        return _commit(
            engine, current_guess, scoring.get_score(current_guess, solution), threshold_display
        )

    weighted = engine.weights is not None
    while True:
        try:
            raw = input(f"Suggested {current_guess}. Command: ")
        except EOFError:
            return LoopState.QUIT

        # Convenience for known-solution runs: an empty line just accepts
        # the current guess and lets the solution resolve its score,
        # rather than requiring a throwaway ShowScore value the solution is
        # going to override anyway.
        if not raw.strip() and solution is not None:
            return _commit(
                engine,
                current_guess,
                scoring.get_score(current_guess, solution),
                threshold_display,
            )

        command = parse_command(raw, engine.n)
        if command is None:
            logging.info(f"Could not understand {raw!r}, ignoring.")
            continue

        if isinstance(command, Restart):
            logging.info("Restart")
            return LoopState.RESTART

        if isinstance(command, Quit):
            logging.info("Quit")
            return LoopState.QUIT

        if isinstance(command, Help):
            sys.stdout.write(_HELP_TEXT + "\n")
            continue

        if isinstance(command, OverrideGuess):
            current_guess = command.word
            logging.info(f"Using {current_guess}")
            continue

        if isinstance(command, Analyze):
            result = engine.analyze(command.word)
            sys.stdout.write(display.format_top_guesses([result], weighted=weighted) + "\n")
            continue

        if isinstance(command, Buckets):
            result = engine.analyze(command.word or current_guess)
            sys.stdout.write(display.format_buckets(result, weights=engine.weights) + "\n")
            continue

        if isinstance(command, Top):
            ranked = engine.get_ranked_analyses()
            sys.stdout.write(
                display.format_top_guesses(ranked, top_n=command.n, weighted=weighted) + "\n"
            )
            continue

        # ShowScore: solution (if known) always overrides a manually typed
        # value, matching the priority the old get_guess_score gave it.
        score = scoring.get_score(current_guess, solution) if solution is not None else command.value
        return _commit(engine, current_guess, score, threshold_display)


def run_interactive(engine, automatic, solution, threshold_display=3):
    state = LoopState.CONTINUE
    while state == LoopState.CONTINUE:
        state = play_one_round(engine, automatic, solution, threshold_display)
        if state == LoopState.RESTART:
            engine.reset()
            state = LoopState.CONTINUE
        elif state in (LoopState.ERROR, LoopState.SOLVED):
            try:
                r = input("New round? ").lower().strip()
            except EOFError:
                r = ""

            if r and r in ("r", "1", "y"):
                logging.info("New round")
                engine.reset()
                state = LoopState.CONTINUE
            else:
                state = LoopState.QUIT


STRATEGIES: dict[str, type[Strategy]] = {
    "entropy": EntropyStrategy,
    "expected-pool-size": ExpectedPoolSizeStrategy,
    "minimax": MinimaxStrategy,
    "two-ply-expectimax": TwoPlyExpectimaxStrategy,
}


def build_strategy(name: str, weighted: bool) -> Strategy:
    cls = STRATEGIES[name]
    if cls is MinimaxStrategy:
        if weighted:
            logging.warning("minimax has no weighted mode; ignoring --weighted")
        return MinimaxStrategy()
    return cls(weighted=weighted)


def run(args):
    word_list = parse_file(args.infile)
    target, extra = word_list.target, word_list.extra
    logging.info(
        f"Target {len(target)} extra {len(extra)} wordlen {word_list.word_length}"
    )

    # Allowed guesses are the full list (target + extra); possible
    # solutions are restricted to the target list.
    guesses = sorted(set(target) | set(extra))
    targets = target

    engine = SolverEngine(
        guesses,
        targets,
        build_strategy(args.strategy, args.weighted),
        weights=word_list.weights,
        initial_guess=args.initial_guess,
    )
    run_interactive(
        engine, args.automatic, args.solution, threshold_display=args.threshold_display
    )


def setup_logging(debug):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=lvl, format="%(message)s")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactively solve Wordle-style puzzles by suggesting "
        "high-information guesses and narrowing the candidate pool as you "
        "report each guess's score.",
    )
    parser.add_argument(
        "infile",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="word list file to read (see wordlists.py for format); "
        "defaults to stdin",
    )
    parser.add_argument(
        "-i",
        "--initial-guess",
        default=None,
        help="force this word as the first guess instead of computing one",
    )
    parser.add_argument(
        "-T",
        "--threshold-display",
        default=3,
        type=int,
        help="print the full list of remaining candidates once the pool "
        "shrinks to this size or smaller (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--solution",
        default=None,
        help="known solution word; scores are computed automatically "
        "instead of prompted for",
    )
    parser.add_argument(
        "-a",
        "--automatic",
        action="store_true",
        help="play without prompting, using -s/--solution to score each "
        "guess (requires --solution)",
    )
    parser.add_argument(
        "-S",
        "--strategy",
        choices=sorted(STRATEGIES),
        default="entropy",
        help="heuristic used to rank candidate guesses (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--weighted",
        action="store_true",
        help="rank guesses using word-frequency weights from the word "
        "list, when available (ignored by minimax)",
    )
    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="enable debug logging",
    )

    args = parser.parse_args(argv)
    setup_logging(args.debug)
    run(args)


if __name__ == "__main__":
    main()
