#!/usr/bin/env python3
"""Benchmark harness for wordle_claude.c.

Runs the solver binary against a set of fixed-size fixture word lists (fast,
for iterating on heuristics.md changes) and optionally the real words.txt
(slow, for realistic numbers), parses its exact-result stats, and appends
labeled results to benchmark_results.jsonl.

`--compare OLD NEW` diffs two labels' latest runs per test case: time and
node-count deltas, plus a hard check that exact_total_guesses is unchanged.
Per wordle_claude.c's own design goals, every optimization must only affect
how fast the optimum is found, never what the optimum is -- so a changed
total is reported as a correctness regression, not a benchmark result.

Usage:
    solver/benchmark_solver.py --build --label before
    ... make a heuristics.md change ...
    solver/benchmark_solver.py --build --label after
    solver/benchmark_solver.py --compare before after
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SOLVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOLVER_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
import wordlists  # noqa: E402

DEFAULT_BINARY = SOLVER_DIR / "wordle_claude"
FIXTURE_DIR = SOLVER_DIR / "bench_wordlists"
RESULTS_FILE = SOLVER_DIR / "benchmark_results.jsonl"
FULL_WORDLIST = REPO_ROOT / "words.txt"

# name -> (num_targets, num_extra_guesses). Sized to stress the solver's
# near-leaf recursion (tiny) up through a real branch-and-bound search
# (medium) while staying fast enough to run on every change.
FIXTURES = {
    "tiny": (12, 0),
    "small": (40, 120),
    "medium": (120, 360),
}

CONVERTERS = {
    "exact_total_guesses": int,
    "avg_guesses": float,
    "solver_time_sec": float,
    "nodes": int,
}

RESULT_PATTERNS = {
    "exact_total_guesses": re.compile(r"Exact Total Guesses:\s*(\d+)"),
    "avg_guesses": re.compile(r"Exact Average Score:\s*([\d.]+)"),
    "solver_time_sec": re.compile(r"Computation Time:\s*([\d.]+)"),
    "nodes": re.compile(r"Tree Nodes Visited:\s*(\d+)"),
}


def build_fixture_wordlist(name, num_targets, num_extra):
    with open(FULL_WORDLIST) as fp:
        wl = wordlists.parse_file(fp)
    targets = wl.target[:num_targets]
    extra = wl.extra[:num_extra]
    if len(targets) < num_targets or len(extra) < num_extra:
        raise ValueError(f"words.txt too small for fixture {name!r}")

    FIXTURE_DIR.mkdir(exist_ok=True)
    path = FIXTURE_DIR / f"{name}.txt"
    with open(path, "w") as fp:
        fp.write("\n".join(targets) + "\n\n")
        fp.write("\n".join(extra) + ("\n" if extra else ""))
    return path


def ensure_fixtures(regen=False):
    paths = {}
    for name, (num_targets, num_extra) in FIXTURES.items():
        path = FIXTURE_DIR / f"{name}.txt"
        if regen or not path.exists():
            path = build_fixture_wordlist(name, num_targets, num_extra)
        paths[name] = path
    return paths


def build_binary(do_build):
    if do_build:
        subprocess.run(["make", "-C", str(SOLVER_DIR)], check=True)
    if not DEFAULT_BINARY.exists():
        raise SystemExit(f"{DEFAULT_BINARY} not found; pass --build")
    return DEFAULT_BINARY


def binary_fingerprint(binary):
    return hashlib.sha256(Path(binary).read_bytes()).hexdigest()[:12]


def run_case(binary, wordlist_path, opener, threads):
    cmd = [str(binary), "--wordlist", str(wordlist_path), "--opener", opener, "--threads", str(threads)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_sec = time.time() - t0

    parsed = {}
    for key, pattern in RESULT_PATTERNS.items():
        m = pattern.search(proc.stdout)
        if not m:
            raise RuntimeError(f"couldn't parse {key!r} from solver output:\n{proc.stdout}")
        parsed[key] = CONVERTERS[key](m.group(1))
    parsed["wall_sec"] = wall_sec
    return parsed


def opener_for(wordlist_path):
    with open(wordlist_path) as fp:
        return wordlists.parse_file(fp).target[0]


def load_results():
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE) as fp:
        return [json.loads(line) for line in fp if line.strip()]


def latest_by_label_case(records):
    latest = {}
    for r in records:
        key = (r["label"], r["case"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    return latest


def compare(old_label, new_label):
    latest = latest_by_label_case(load_results())
    cases = sorted({case for (label, case) in latest if label in (old_label, new_label)})
    if not cases:
        raise SystemExit(f"no results found for labels {old_label!r}/{new_label!r}")

    header = f"{'case':<8} {'total (old->new)':<20} {'time (old->new)':<24} {'speedup':>8}   {'nodes (old->new)':<24} {'reduction':>9}"
    print(header)
    print("-" * len(header))

    correctness_ok = True
    for case in cases:
        o = latest.get((old_label, case))
        n = latest.get((new_label, case))
        if not o or not n:
            print(f"{case:<8} missing a result for one of the labels, skipping")
            continue

        total_match = o["exact_total_guesses"] == n["exact_total_guesses"]
        correctness_ok &= total_match
        total_str = f"{o['exact_total_guesses']}->{n['exact_total_guesses']}"
        if not total_match:
            total_str += "  !!"

        speedup_str = f"{o['solver_time_sec'] / n['solver_time_sec']:6.2f}x" if n["solver_time_sec"] else "   n/a "
        node_reduction = 1 - (n["nodes"] / o["nodes"]) if o["nodes"] else 0.0

        print(
            f"{case:<8} {total_str:<20} "
            f"{o['solver_time_sec']:.3f}s->{n['solver_time_sec']:.3f}s".ljust(len('time (old->new)') + 9)
            + f" {speedup_str}   "
            f"{o['nodes']}->{n['nodes']}".ljust(24)
            + f" {node_reduction * 100:7.1f}%"
        )

    if not correctness_ok:
        print(
            "\nFAIL: exact_total_guesses differs between labels on at least one case.\n"
            "Per wordle_claude.c's design goals, any change here is a correctness\n"
            "regression, not a speed improvement -- do not treat it as one."
        )
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--build", action="store_true", help="run `make -C solver` before benchmarking")
    parser.add_argument("--regen-wordlists", action="store_true", help="rebuild the fixture word lists from words.txt")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[*FIXTURES, "full"],
        default=list(FIXTURES),
        help="which fixtures to run; 'full' uses the real words.txt (slow) (default: %(default)s)",
    )
    parser.add_argument("--full-opener", default="salet", help="opener word for the 'full' case (default: %(default)s)")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="solver thread count; kept at 1 by default so node counts are deterministic and comparable across runs (default: %(default)s)",
    )
    parser.add_argument("--label", default=None, help="tag for this run's results (default: sha256 prefix of the binary)")
    parser.add_argument("--compare", nargs=2, metavar=("OLD_LABEL", "NEW_LABEL"), help="diff two labels' latest results instead of running")
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return

    binary = build_binary(args.build) if args.binary == str(DEFAULT_BINARY) else Path(args.binary)
    if not binary.exists():
        raise SystemExit(f"{binary} not found")

    fixtures = ensure_fixtures(regen=args.regen_wordlists) if any(c != "full" for c in args.cases) else {}
    label = args.label or binary_fingerprint(binary)

    results = []
    for case in args.cases:
        wordlist_path = FULL_WORDLIST if case == "full" else fixtures[case]
        opener = args.full_opener if case == "full" else opener_for(wordlist_path)

        print(f"running {case} (opener={opener})...", file=sys.stderr)
        parsed = run_case(binary, wordlist_path, opener, args.threads)
        record = {"label": label, "case": case, "opener": opener, "timestamp": time.time(), **parsed}
        results.append(record)
        print(
            f"  total={parsed['exact_total_guesses']} avg={parsed['avg_guesses']:.5f} "
            f"time={parsed['solver_time_sec']:.3f}s nodes={parsed['nodes']}",
            file=sys.stderr,
        )

    with open(RESULTS_FILE, "a") as fp:
        for r in results:
            fp.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(results)} result(s) to {RESULTS_FILE} under label {label!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
