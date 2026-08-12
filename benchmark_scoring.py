"""Benchmark scoring performance across implementations and matrix sizes.

Compares:
  1. Pure Python scalar (scoring.get_score)
  2. NumPy vectorized batch scoring (_score_batch_numpy)
  3. C-accelerated multithreaded scoring (_score_c)
  4. Fused C score + bincount computation
"""

import argparse
import sys
import time

import numpy as np

import analysis
import fast_scoring
import scoring
from wordlists import parse_file


def benchmark_scalar(guesses, targets, n_samples=10000):
    G, T = len(guesses), len(targets)
    t0 = time.perf_counter()
    count = 0
    for i in range(n_samples):
        g = guesses[i % G]
        t = targets[i % T]
        scoring.get_score(g, t)
        count += 1
    t1 = time.perf_counter()
    elapsed = t1 - t0
    rate = count / elapsed / 1e6
    print(f"  Scalar Python:         {elapsed*1000:8.2f} ms for {count:,} pairs ({rate:6.2f} M pairs/sec)")
    return elapsed


def benchmark_numpy(guesses, targets):
    G, T = len(guesses), len(targets)
    total_pairs = G * T
    t0 = time.perf_counter()
    fast_scoring._score_matrix_numpy(guesses, targets)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    rate = total_pairs / elapsed / 1e6
    print(f"  NumPy Vectorized:      {elapsed:8.4f} s  ({total_pairs:,} pairs, {rate:6.2f} M pairs/sec)")
    return elapsed


def benchmark_c(guesses, targets):
    if not fast_scoring.HAS_C_LIB:
        print("  C Acceleration:        N/A (C library not available/compiled)")
        return None
    G, T = len(guesses), len(targets)
    total_pairs = G * T
    t0 = time.perf_counter()
    fast_scoring._score_matrix_c(guesses, targets)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    rate = total_pairs / elapsed / 1e6
    print(f"  C Accelerated:         {elapsed:8.4f} s  ({total_pairs:,} pairs, {rate:6.2f} M pairs/sec)")
    return elapsed


def benchmark_fused(guesses, targets, weights_dict=None):
    G, T = len(guesses), len(targets)
    total_pairs = G * T

    t0 = time.perf_counter()
    matrix, counts, masses = fast_scoring.score_matrix_and_bincounts(
        guesses, targets, weights=weights_dict
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    rate = total_pairs / elapsed / 1e6
    impl = "C Fused" if fast_scoring.HAS_C_LIB else "NumPy + Bincount"
    print(f"  {impl:22s} {elapsed:8.4f} s  ({total_pairs:,} pairs, {rate:6.2f} M pairs/sec)")
    return elapsed


def benchmark_analyze_all(guesses, targets, weights_dict=None, sizes=(5, 50, 500)):
    """Times analysis.analyze_all -- the actual function SolverEngine calls
    every round -- at realistic candidate-pool sizes with the *full* guess
    list held fixed. This is deliberately different from the raw
    score_matrix benchmarks above: G (the guess list) never shrinks during a
    real game, only T (the candidate pool) does, and analyze_all's
    per-guess reduction and GuessAnalysis construction cost scales with G,
    not T. So a small T (a near-solved pool, the common case after round 1)
    is not a small workload here the way it is for score_matrix -- it's
    dominated by fixed per-guess overhead, which the raw scoring benchmarks
    above don't show at all.
    """
    G = len(guesses)
    for T in sizes:
        pool = targets[: min(T, len(targets))]
        analysis.analyze_all(guesses, pool, weights=weights_dict)  # warm
        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            analysis.analyze_all(guesses, pool, weights=weights_dict)
        elapsed = (time.perf_counter() - t0) / reps
        print(f"  T={len(pool):5d} (G={G:,} fixed):  {elapsed*1000:8.3f} ms/round")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", nargs="?", default="words.wordle.txt")
    args = parser.parse_args(argv)

    with open(args.infile) as fp:
        wl = parse_file(fp)
    guesses = sorted(set(wl.target) | set(wl.extra))
    targets = wl.target

    print(f"Loaded {len(guesses):,} guesses and {len(targets):,} targets from {args.infile}")
    print(f"C extension active: {fast_scoring.HAS_C_LIB}\n")

    print("=== 1. Scalar Scoring Benchmark ===")
    benchmark_scalar(guesses, targets)
    print()

    print(f"=== 2. Matrix Scoring (Round 1 Pool: {len(guesses):,} x {len(targets):,} = {len(guesses)*len(targets):,} pairs) ===")
    t_np = benchmark_numpy(guesses, targets)
    t_c = benchmark_c(guesses, targets)
    if t_c and t_np:
        print(f"  Speedup:               {t_np / t_c:.1f}x faster with C acceleration")
    print()

    print(f"=== 3. Full Matrix Scoring ({len(guesses):,} x {len(guesses):,} = {len(guesses)**2:,} pairs) ===")
    t_np_full = benchmark_numpy(guesses, guesses)
    t_c_full = benchmark_c(guesses, guesses)
    if t_c_full and t_np_full:
        print(f"  Speedup:               {t_np_full / t_c_full:.1f}x faster with C acceleration")
    print()

    print("=== 4. Fused Matrix + Bincount Benchmark ===")
    benchmark_fused(guesses, targets, wl.weights)
    print()

    print("=== 5. analyze_all at realistic in-game pool sizes (full guess list held fixed) ===")
    benchmark_analyze_all(guesses, targets, wl.weights, sizes=(5, 50, 500, len(targets)))
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
