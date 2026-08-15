#!/usr/bin/env python3
"""Test harness for wordle_gemini.c: correctness regressions, sanitizer
issues (ASan/UBSan memory bugs, TSan data races).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SOLVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOLVER_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "solver"))
import scoring  # noqa: E402
import wordlists  # noqa: E402
import reference_solver  # noqa: E402

EXACT_MATCH = 242

VARIANTS = {
    "plain": {"binary": SOLVER_DIR / "wordle_gemini", "make_target": "all", "env": {}},
    "asan": {
        "binary": SOLVER_DIR / "wordle_gemini_asan",
        "make_target": "asan",
        "env": {"ASAN_OPTIONS": "halt_on_error=1:abort_on_error=1", "UBSAN_OPTIONS": "halt_on_error=1"},
    },
    "tsan": {
        "binary": SOLVER_DIR / "wordle_gemini_tsan",
        "make_target": "tsan",
        "env": {"TSAN_OPTIONS": "halt_on_error=1"},
    },
}

SANITIZER_MARKERS = re.compile(
    r"ERROR: (?:Address|Leak)Sanitizer|WARNING: ThreadSanitizer|runtime error:|SUMMARY: \w*Sanitizer"
)

ORACLE_CASES = {
    "disjoint_letters": ["abcde", "fghij", "klmno", "pqrst", "uvwxy"],
    "duplicate_letters": ["sassy", "mamma", "gaffe", "eerie", "esses", "sheer"],
    "real_words_small": None,
}

STRESS_CASES = ["small", "medium"]
FIXTURE_DIR = REPO_ROOT / "solver" / "bench_wordlists"
FULL_WORDLIST = REPO_ROOT / "words.txt"
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


def load_real_words_small(n=7):
    with open(FULL_WORDLIST) as fp:
        wl = wordlists.parse_file(fp)
    return wl.target[:n]


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


class SolverFailure(Exception):
    pass


def run_solver(binary, env_extra, args, timeout=120):
    env = {**os.environ, **env_extra}
    proc = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=timeout, env=env)
    combined = proc.stdout + proc.stderr
    marker = SANITIZER_MARKERS.search(combined)
    if marker:
        raise SolverFailure(f"sanitizer flagged {args!r}: {marker.group(0)!r}\n--- output (tail) ---\n{combined[-4000:]}")
    if proc.returncode != 0:
        raise SolverFailure(f"exit {proc.returncode} on {args!r}\n--- output (tail) ---\n{combined[-4000:]}")
    return combined


def parse_single_opener(output):
    parsed = {}
    for key, pattern in RESULT_PATTERNS.items():
        m = pattern.search(output)
        if not m:
            raise SolverFailure(f"couldn't parse {key!r} from single-opener output:\n{output[-2000:]}")
        parsed[key] = CONVERTERS[key](m.group(1))
    return parsed


def parse_search_summary(output):
    m = re.search(r"with exact average score: [\d.]+ \((\d+) total guesses\)", output)
    if not m:
        raise SolverFailure(f"couldn't parse total guesses from --all/--top summary:\n{output[-2000:]}")
    return int(m.group(1))


def tree_guess_count(root, target):
    node = root
    guesses = 0
    while True:
        guesses += 1
        if node.get("leaf"):
            if node["guess"] != target:
                raise SolverFailure(f"leaf {node['guess']!r} reached while resolving target {target!r}")
            return guesses
        s = scoring.get_score(node["guess"], target)
        if s == EXACT_MATCH:
            return guesses
        child = node.get("branches", {}).get(str(s))
        if child is None:
            raise SolverFailure(f"tree has no branch for score {s} at guess {node['guess']!r} (target {target!r})")
        node = child


def verify_tree(tree_path, targets, expected_total):
    with open(tree_path) as fp:
        root = json.load(fp)["tree"]
    total = sum(tree_guess_count(root, t) for t in targets)
    if total != expected_total:
        raise SolverFailure(f"tree replay total {total} != solver-reported exact_total_guesses {expected_total}")


def write_wordlist(path, targets):
    with open(path, "w") as fp:
        fp.write("\n".join(targets) + "\n\n")


def run_oracle_case(name, targets, variant_name, binary, env_extra, tmpdir):
    wordlist_path = Path(tmpdir) / f"{name}.txt"
    write_wordlist(wordlist_path, targets)
    opener = targets[0]

    tree_path = Path(tmpdir) / f"{name}_{variant_name}_tree.json"
    out = run_solver(
        binary,
        env_extra,
        ["--wordlist", str(wordlist_path), "--opener", opener, "--threads", "2", "--tree", str(tree_path)],
    )
    parsed = parse_single_opener(out)
    expected = reference_solver.forced_opener_cost(targets, targets, opener)
    if parsed["exact_total_guesses"] != expected:
        raise SolverFailure(f"--opener {opener}: solver={parsed['exact_total_guesses']} oracle={expected}")
    verify_tree(tree_path, targets, parsed["exact_total_guesses"])

    out = run_solver(
        binary, env_extra, ["--wordlist", str(wordlist_path), "--all", "--threads", "2", "--quiet"]
    )
    total = parse_search_summary(out)
    expected_all = reference_solver.exact_cost(targets, targets)
    if total != expected_all:
        raise SolverFailure(f"--all: solver={total} oracle={expected_all}")


def run_stress_case(case_name, fixture_path, variant_name, binary, env_extra, stress_results):
    with open(fixture_path) as fp:
        opener = wordlists.parse_file(fp).target[0]

    out = run_solver(binary, env_extra, ["--wordlist", str(fixture_path), "--opener", opener, "--threads", "4"])
    opener_total = parse_single_opener(out)["exact_total_guesses"]

    out2 = run_solver(
        binary, env_extra, ["--wordlist", str(fixture_path), "--top", "8", "--threads", "4", "--quiet"]
    )
    top8_total = parse_search_summary(out2)

    stress_results.setdefault((case_name, "opener"), {})[variant_name] = opener_total
    stress_results.setdefault((case_name, "top8"), {})[variant_name] = top8_total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_build:
        for name in args.variants:
            target = VARIANTS[name]["make_target"]
            print(f"building {name} ({target})...", file=sys.stderr)
            subprocess.run(["make", "-C", str(SOLVER_DIR), target], check=True)

    for name in args.variants:
        binary = VARIANTS[name]["binary"]
        if not binary.exists():
            raise SystemExit(f"{binary} missing; run without --skip-build first")

    ORACLE_CASES["real_words_small"] = load_real_words_small()
    fixtures = ensure_fixtures()

    failures = []
    stress_results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for variant_name in args.variants:
            binary = VARIANTS[variant_name]["binary"]
            env_extra = VARIANTS[variant_name]["env"]
            print(f"\n=== {variant_name} ===")

            for case_name, targets in ORACLE_CASES.items():
                print(f"  oracle: {case_name}...", end=" ", flush=True)
                try:
                    run_oracle_case(case_name, targets, variant_name, binary, env_extra, tmpdir)
                    print("ok")
                except SolverFailure as e:
                    print("FAIL")
                    failures.append(f"[{variant_name}] oracle/{case_name}: {e}")

            for case_name in STRESS_CASES:
                print(f"  stress: {case_name}...", end=" ", flush=True)
                try:
                    run_stress_case(case_name, fixtures[case_name], variant_name, binary, env_extra, stress_results)
                    print("ok")
                except SolverFailure as e:
                    print("FAIL")
                    failures.append(f"[{variant_name}] stress/{case_name}: {e}")

    for (case_name, mode), by_variant in stress_results.items():
        if len(set(by_variant.values())) > 1:
            failures.append(f"cross-build mismatch on stress/{case_name}/{mode}: {by_variant}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
