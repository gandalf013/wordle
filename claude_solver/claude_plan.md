# Plan: closing the speed gap with alex1770/wordle.cpp — claude_solver/

Supersedes `solver/claude_plan.md` for ongoing work in this directory
(that file is left untouched per instruction not to modify anything under
`solver/`). See `PROGRESS.md` in this directory for full detail on what's
been done and the measured numbers behind each line below.

## Status

**Phase 1 (items 1, 2, 3, 4, 5): done, verified, committed.**
**Phase 2 (re-measurement): done.**
Not started: item 6 (loop fusion), Phase 3 (disjoint-union bound
propagation), Phase 4 (endgame-cluster cutoff).

Headline result: single-threaded, full word list, salet opener —
864.4s (pre-Phase-1) → **405.1s (post-Phase-1)**, a real 2.1x speedup,
`exact_total_guesses` unchanged (11433) throughout. Gap to `wordle.cpp`'s
34.25s reference narrowed from ~25x to ~11.8x. Meaningful, but Phase 1's
cheap wins did not close most of the gap — Phase 4 remains warranted.

## Scope (unchanged)

- Easy mode only. No item depends on `wordle.cpp`'s `oktestwords`-style
  hard-mode bookkeeping.
- Goal: make `--opener` and `--top`/`--all` usably fast on the real word
  list, not feature parity with `wordle.cpp`.

## Next steps, in order

### 1. Item 6 — inner-loop fusion (deferred from Phase 1)

Fuse the dedup pass, move-ordering pass, and the main loop's own
histogram rebuild into fewer scans over `targets`. Motivated by both this
session's profiling (~13% of time in `qsort`/ranking overhead) and
`findings.md`'s framing (three separate `O(#guesses × #targets)` passes
per node). Pure refactoring, no algorithmic change — but merging loops
carries more risk of a silent off-by-one than Phase 1's additive changes,
so: implement it as its own isolated change, verify with the full
`test_solver.py` and a `benchmark_solver.py --compare` before moving on,
same as every other item so far.

### 2. Phase 3 — disjoint-union bound propagation

`heuristics.md` #3, using the cleaner standalone formulation (not
`wordle.cpp`'s version, which is entangled with its endgame-tracking
state): when a guess's bucket-by-bucket search fails against beta partway
through, the buckets *already solved* before the failure point have real,
established costs. Since sibling buckets from the same guess are disjoint
subsets of the same parent, `cost(A ∪ B) ≥ cost(A) + cost(B)` (provable:
restricting an optimal A∪B strategy to A-only outcomes gives a valid, if
not necessarily optimal, strategy for A alone, so its cost is ≥ the true
A-optimum; same for B; sum). Store `L_A + L_B` as a lower bound for the
union `A ∪ B`, keyed by `hash1(A) XOR hash1(B)` / `hash2(A) XOR hash2(B)`
(free — zobrist hashes are already tracked per-bucket) and
`size(A) + size(B)`, for any sibling search that later hits that exact
union as its own subset. No endgame infrastructure required — implement
and test this on its own before Phase 4.

**Design questions to resolve before writing code** (flagging now so
they're not skipped under momentum from Phase 1's smooth run):
- Where exactly to hook the union-bound write-in: right after a bucket
  loop's `pruned = true` break, using whichever buckets were already
  solved in that iteration.
- Whether to write via `tt_find_or_claim` directly (bypassing
  `solve_subset`'s normal entry path) — needs its own explicit
  correctness argument for interacting with the existing
  `proven_lower_bound` semantics (should only *raise* a bound, never
  overwrite a tighter existing one or an `exact_cost`).
- Verify TT capacity assumptions still hold — this adds new entries for
  subsets that were never directly *searched*, only *derived*, so the
  table may fill faster than before.

### 3. Phase 4 — endgame-cluster cutoff (the big one)

`heuristics.md` #2. Scope the *static* coverage cutoff only
(`wordle.cpp:685-709`'s greedy coverage-counting bound), without the
live-endgame fallback re-search (`wordle.cpp:715-721`) — the fallback is
a second solve over a different subset framing sharing the same
recursion/cache machinery, exactly the kind of change that risks silently
conflating two problems under one TT key. Needs its own soundness
write-up before implementation, same standard items 6-8 and the Phase 1
items were held to.

Correctness notes already banked from reviewing `gemini_plan.md`'s code
sketch of this (carry forward, don't relitigate):
- Do not return a raw `UINT32_MAX` sentinel for "no solution possible" —
  it collides with the TT's existing "exact cost not yet computed"
  sentinel (`exact_cost == UINT32_MAX`). Fail soft against the caller's
  `beta`, matching every other cutoff in `solve_subset`.
- Don't copy `wordle.cpp`'s gating conditions (or gemini's simplified
  `count >= 4` version) without re-deriving why they're the right
  threshold for *this* solver's specific bound formulas and objective
  (total/average cost, not depth-limited feasibility).

### 4. Re-measure again after Phase 3/4

Same salet/words.txt/wordle.cpp comparison as Phase 2, to see how much of
the remaining ~11.8x gap each phase actually closes in practice, not just
in theory.

## Explicit non-goals (unchanged from solver/claude_plan.md)

- Hard/ultrahard mode.
- `--all` as a routine operation (still likely intractable without the
  endgame work landing broadly).
- GPU acceleration (analyzed and ruled out earlier this session).
- Disk-persisted cache.
- Global shared lock-free TT (no profiling evidence motivates it; would
  contradict `findings.md`'s own measured TT health stats).
- Seeding beta for `--top`/`--all`'s initial ceiling or for `--tree`
  dump's build path — both still use `UINT32_MAX` unconditionally. Real,
  same-shaped opportunity as item 2, deliberately left out of Phase 1 to
  keep that pass scoped; worth a small follow-up but not urgent.

## Verification approach (unchanged)

Full `test_solver.py` (independent oracle + ASan + UBSan + TSan +
cross-build stress agreement) after every change.
`benchmark_solver.py --compare` with labeled before/after runs. Every
sub-technique in Phase 3/4 gets its own explicit correctness argument
before landing.
