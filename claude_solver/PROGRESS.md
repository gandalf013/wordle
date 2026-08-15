# Progress against solver/claude_plan.md

Tracking status here (inside `claude_solver/`) rather than editing
`solver/claude_plan.md`, per instruction not to touch anything in `solver/`.

## Step 0 — RESOLVED

Ran `--top 5 --threads 4` on the real word list with the (then-unmodified)
baseline solver: **307.57s** (5:07.57 wall, 382% CPU utilization — good
parallelism this time, not a throttling artifact). This does **not**
reproduce `findings.md`'s claimed 166s, and matches my own prior
extrapolation from the `--top 25` run. `findings.md`'s "`--top`/`--all`
already ~20x more efficient than `--opener`" framing does not hold up under
direct reproduction — treat that framing as unverified/likely wrong going
forward, not as an established fact.

## Phase 1 — DONE (items 1, 2, 3, 4, 5); item 6 not yet done

All changes made only in `claude_solver/wordle_claude.c` (copy of
`solver/wordle_claude.c` as of the start of this work, i.e. including
`heuristics.md` items 6-8 already). Verified after every change with the
full `test_solver.py` (independent oracle + ASan + UBSan + TSan +
cross-build stress agreement) — all green throughout.

1. **DONE** — node-count reporting bug fixed (`bucket_worker` now captures
   a per-bucket delta via `nodes_before`, instead of writing the
   cumulative per-thread counter into every bucket's slot). Verified:
   `--threads 1` and `--threads 4` now report identical node counts for
   the same opener/fixture (120 both ways on the `medium` fixture),
   whereas before this fix they diverged based on thread count.
2. **DONE** — beta seeding for `--opener`. Added `greedy_pick` +
   `greedy_upper_bound`, which play out a real (not estimated) greedy
   strategy to get an achievable upper bound, then derive each root
   bucket's own beta from it (mirroring the existing ceiling-derivation
   pattern already used in `evaluate_opener_sequential` for `--top`/
   `--all`). Correctness argument written inline in
   `evaluate_opener_parallel`: every bucket's derived beta is proven `>=`
   that bucket's own true cost, so `solve_subset`'s "exact if returned
   value < beta" contract still holds (with the same at-the-boundary tie
   caveat that already exists elsewhere in this codebase's beta
   conventions — doesn't affect the returned total, only whether a tied
   bucket's TT entry is marked "exact" vs. "proven lower bound"). Scoped
   to `--opener` only, not `--top`/`--all`'s ceiling or `--tree` dump
   (both still start at/use `UINT32_MAX`) — a possible follow-up, not
   done here.
3. **DONE** — `lb1` node-level fail-soft bound, folded into the existing
   ranking pass (not a new full-`num_guesses` pass): each representative's
   own `guess_lb` is already computed there for `sum_sq`/`active_buckets`;
   tracking `lb1 = min` over those and updating `node_lb` before `qsort`
   lets a node fail soft without ever reaching the main candidate loop.
   Correct because dedup guarantees every non-representative guess shares
   its representative's exact histogram (same `glb`), so `min` over
   representatives equals `min` over all `num_guesses` guesses.
4. **DONE (no-op)** — `heuristics.md` #13 (smallest-first bucket order)
   confirmed dropped from consideration; no code exists for it and none
   was added.
5. **DONE** — double-hashing. Added `zobrist2` (independent splitmix64
   stream), `TTEntry.hash1`/`hash2`, `BucketInfo.hash1`/`hash2`,
   `Solver.dedup_hash2`, and a second independent rolling hash (different
   offset basis and multiplier) computed alongside the existing one in the
   dedup pass. `tt_find`/`tt_find_or_claim` and the dedup duplicate-check
   now require both hashes (plus size) to match. Touched every call site
   in the file (`solve_subset`, `partition_root`,
   `evaluate_opener_sequential`, `bucket_worker`,
   `build_subtree_node`/`build_subtree_node_with_guess`/
   `tree_bucket_worker`) — genuinely invasive, but mechanical (threading
   an extra value through, no logic changes beyond the comparisons
   themselves). Clean rebuild with zero warnings; full harness green
   after.
6. **NOT DONE** — inner-loop fusion (dedup + move-ordering + sum_sq into
   one pass). Deferred; the plan called for doing this last/in isolation
   given higher risk of a silent off-by-one from merging three loops, and
   items 1-3-5 already delivered a measurable win (see below) worth
   checkpointing on before taking on more risk in the same pass.

## Phase 2 — re-measurement (in progress)

Full word list, `--opener salet --threads 4`: **157.577s**, total
**11433** unchanged (correctness preserved), node count now trustworthy
(reporting bug fixed): 2,251,108. Down from the pre-Phase-1 baseline of
**235.006s** (same opener, same word list, `--threads 4`, before any of
items 6-8 or this phase's work) — roughly a 33% wall-clock reduction,
though this machine has demonstrated real run-to-run timing noise on long
multi-threaded runs, so treat this as directional, not precise.

Clean **single-threaded** run (directly comparable to `wordle.cpp`'s own
single-core 34.25s reference on this exact word list): **405.098s**, total
**11433** unchanged. Node count matches the 4-threaded run exactly
(2,251,108 both ways) — further confirmation the node-count-reporting fix
works correctly at full scale, not just on fixtures.

Before/after summary (single-threaded, full word list, salet):
- Pre-Phase-1 (items 6-8 only): 864.431s
- Post-Phase-1 (items 1,2,3,5): 405.098s -- **~2.1x speedup, correctness
  unchanged**
- `wordle.cpp` reference: 34.25s

Gap to `wordle.cpp` narrowed from ~25x to ~11.8x. Real, meaningful progress,
but Phase 1's cheap wins did **not** close most of the gap the way
`findings.md`'s ">300s to tens of seconds" estimate for beta-seeding
implied at full-word-list scale -- that estimate looks optimistic here.
This means Phase 4 (endgame-cluster cutoff) is still genuinely warranted,
not obviated by the cheap wins -- consistent with treating the Phase 2
re-measurement as a real decision point rather than a formality.

## Status: paused here for a checkpoint

Phase 1 (items 1,2,3,4,5) done and verified. Item 6 (loop fusion), Phase 3
(disjoint-union bound propagation), and Phase 4 (endgame-cluster cutoff)
not yet started. Given the re-measured gap still warrants them, and Phase
4 in particular is the highest-risk item in the whole plan, pausing here
to report back before continuing further.
