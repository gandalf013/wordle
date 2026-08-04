"""cli.py: the interactive REPL and CLI entry point built on SolverEngine.
The only module in the wordle solver with input()/print()/logging calls --
scoring, analysis, and strategy selection are all pure by this point, so
this module's only job is deciding how to surface them interactively.

The round-by-round flow here (play_one_round/run_interactive) mirrors the
old Game.play_one_round/run() exactly, just reading suggestions from
SolverEngine.suggest() instead of Game.find_best_guess() and committing via
SolverEngine.apply_score() instead of mutating Game.target_lists directly.

One feature was deliberately dropped rather than ported: the old
-n/--num-top-guesses flag, which logged the top N guesses by entropy on
every round. Reproducing it here would mean duplicating
SolverEngine.suggest()'s internal branching (initial-guess override, cached
first-round suggestion) just to peek at the ranked list one level down --
it's superseded by a proper `top` REPL command in a later step instead of
being carried forward as a CLI flag.
"""

import argparse
import logging
import sys
from enum import Enum, auto

import scoring
from engine import RoundOutcome, SolverEngine
from strategies import EntropyStrategy
from wordlists import parse_file


class LoopState(Enum):
    QUIT = auto()
    RESTART = auto()
    CONTINUE = auto()
    ERROR = auto()
    SOLVED = auto()


def resolve_score(engine, guess, potential_score, solution):
    """Resolve the actual score for `guess` this round: a known solution
    wins outright, then an already-typed candidate score string, then an
    explicit prompt loop. Returns (LoopState, score)."""
    if solution is not None:
        return LoopState.CONTINUE, scoring.get_score(guess, solution)

    if potential_score:
        if len(potential_score) != engine.n:
            logging.info(
                f"Score {potential_score!r} is not {engine.n} characters, ignoring."
            )
        else:
            try:
                potential_score = int(potential_score, 3)
            except (TypeError, ValueError):
                logging.info(f"Could not understand {potential_score}, ignoring.")
            else:
                return LoopState.CONTINUE, potential_score

    s = ""
    while len(s) != engine.n:
        try:
            s = input(f"Enter score for '{guess}': ").strip()
        except EOFError:
            logging.info("Quit")
            return LoopState.QUIT, None

        if not s:
            continue

        if s[0].lower() == "r":
            logging.info("Restart")
            return LoopState.RESTART, None

        if s[0].lower() == "q":
            logging.info("Quit")
            return LoopState.QUIT, None

    return LoopState.CONTINUE, int(s, base=3)


def play_one_round(engine, automatic, solution, threshold_display=3):
    suggestion = engine.suggest()
    best_guess = suggestion.guess
    logging.info(f"Best guess {best_guess} entropy {suggestion.entropy}")

    new_input = ""
    if not automatic:
        try:
            new_input = input(f"Suggested {best_guess}. Score/new guess: ")
        except EOFError:
            return LoopState.QUIT
        new_input = new_input.lower().strip()

    potential_score = new_input
    if new_input and len(new_input) == engine.n and new_input.islower():
        logging.info(f"Using {new_input} instead of {best_guess}")
        best_guess = new_input
        potential_score = None

    state, guess_score = resolve_score(engine, best_guess, potential_score, solution)
    if guess_score is not None:
        sys.stdout.write(f"{scoring.get_score_str(guess_score, engine.n)}\n")

    if state != LoopState.CONTINUE:
        return state

    result = engine.apply_score(best_guess, guess_score)
    if result.outcome == RoundOutcome.ERROR:
        logging.error("No guess matches the score!")
        return LoopState.ERROR

    if result.outcome == RoundOutcome.SOLVED:
        logging.info(f"SOLVED: {result.solution} in {len(engine.history)} guesses")
        for guess, score in engine.history:
            sys.stdout.write(f"{guess} {scoring.get_score_str(score, engine.n)}\n")
        return LoopState.SOLVED

    logging.info(f"{result.candidates_remaining} words match the pattern")
    if result.candidates_remaining <= threshold_display:
        logging.info(f"Matching words: {sorted(engine.candidates)}")
    logging.debug(f"Words:\n{engine.candidates}")

    return LoopState.CONTINUE


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


def run(args):
    word_list = parse_file(args.infile)
    target, extra = word_list.target, word_list.extra
    logging.info(
        f"Target {len(target)} extra {len(extra)} wordlen {word_list.word_length}"
    )
    words = sorted(set(target) | set(extra))

    if args.guesses == "all":
        guesses = words
    elif args.guesses == "target":
        guesses = target
    elif args.guesses == "extra":
        guesses = extra
    else:
        raise ValueError(f"Unknown 'guesses': {args.guesses}")

    if args.targets == "all":
        targets = words
    elif args.targets == "target":
        targets = target
    elif args.targets == "extra":
        targets = extra
    else:
        raise ValueError(f"Unknown 'targets': {args.targets}")

    engine = SolverEngine(
        guesses,
        targets,
        EntropyStrategy(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action="store_true")
    parser.add_argument(
        "infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument("-g", "--guesses", default="all")
    parser.add_argument("-t", "--targets", default="target")
    parser.add_argument("-i", "--initial-guess", default=None)
    parser.add_argument("-T", "--threshold-display", default=3, type=int)
    parser.add_argument("-s", "--solution", default=None)
    parser.add_argument("-a", "--automatic", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(args.debug)
    run(args)


if __name__ == "__main__":
    main()
