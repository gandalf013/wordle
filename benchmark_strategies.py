"""Compare 1-ply strategies against TwoPlyExpectimaxStrategy by playing out games
against a sample of target solutions.

Evaluates:
  - Simple average guesses
  - Weighted average guesses (using word-frequency weights)
  - Worst-case guess count
  - Games exceeding 6 guesses (fails)
  - Execution time & speed per game
"""

import argparse
import random
import sys
import time

import scoring
from engine import RoundOutcome, SolverEngine
from strategies import EntropyStrategy, ExpectedPoolSizeStrategy, TwoPlyExpectimaxStrategy
from wordlists import parse_file


def play(engine: SolverEngine, solution: str, max_guesses: int = 12) -> int:
    engine.reset()
    for _ in range(1, max_guesses + 1):
        suggestion = engine.suggest()
        score = scoring.get_score(suggestion.guess, solution)
        result = engine.apply_score(suggestion.guess, score)
        if result.outcome == RoundOutcome.SOLVED:
            # RoundResult.guesses_used is the single source of truth: it
            # includes the implied final guess when the pool collapsed to a
            # single candidate without the last played guess being it.
            return result.guesses_used
        if result.outcome == RoundOutcome.ERROR:
            raise RuntimeError(f"no candidate matched while solving for {solution!r}")
    return max_guesses + 1  # didn't converge -- counts as a blown budget


def run_benchmark(name, strategy, guesses, targets, weights, sample):
    engine = SolverEngine(guesses, targets, strategy, weights=weights)
    total_mass = sum(weights.get(w, 1.0) for w in sample)

    t0 = time.time()
    counts = [play(engine, solution) for solution in sample]
    elapsed = time.time() - t0

    weighted_avg = sum(weights.get(w, 1.0) * c for w, c in zip(sample, counts)) / total_mass
    simple_avg = sum(counts) / len(counts)
    worst = max(counts)
    over_six = sum(1 for c in counts if c > 6)

    print(f"=== {name} ===")
    print(f"  games:                {len(sample)}")
    print(f"  simple avg guesses:   {simple_avg:.4f}")
    print(f"  weighted avg guesses: {weighted_avg:.4f}")
    print(f"  worst case:           {worst}")
    print(f"  games over 6 guesses: {over_six}")
    print(f"  wall clock:           {elapsed:.2f}s ({elapsed / len(sample):.3f}s/game)")
    print()
    return weighted_avg, worst, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", nargs="?", default="words.weighted.txt")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="use every target word")
    args = parser.parse_args(argv)

    with open(args.infile) as fp:
        wl = parse_file(fp)
    guesses = sorted(set(wl.target) | set(wl.extra))
    targets = wl.target
    weights = wl.weights

    rng = random.Random(args.seed)
    sample = targets if args.full else rng.sample(targets, min(args.sample_size, len(targets)))

    print(f"guesses={len(guesses)} targets={len(targets)} sample={len(sample)}\n")

    run_benchmark(
        "1-Ply EntropyStrategy(weighted=True)",
        EntropyStrategy(weighted=True),
        guesses,
        targets,
        weights,
        sample,
    )
    run_benchmark(
        "1-Ply ExpectedPoolSizeStrategy(weighted=True)",
        ExpectedPoolSizeStrategy(weighted=True),
        guesses,
        targets,
        weights,
        sample,
    )
    run_benchmark(
        "2-Ply TwoPlyExpectimaxStrategy(weighted=True)",
        TwoPlyExpectimaxStrategy(weighted=True),
        guesses,
        targets,
        weights,
        sample,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
