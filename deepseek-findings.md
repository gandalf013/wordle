# Deep Review: `gemini_solver/wordle_gemini.c`

This is a genuinely strong solver — a fail-soft alpha-beta minimax over the full easy-mode decision tree with a lock-free shared TT, monotone-LB pruning, and a packed single-sort move-ordering scheme. I profiled it (Apple Silicon, clang `-O3 -march=native -flto`), reviewed every branch of `solve_subset` for soundness, and landed three real, measured optimizations. All correctness verified against the known optimum `taler = 11483`, 7 opener differentials vs the old binary, the full oracle/stress/sanitizer suite, and a JSON-tree replay.

## What I verified as sound (you'll want to keep these)

- **`bad==1` pre-check cost `2n`** (`wordle_gemini.c:688`): correct. For count `n`, a target split with exactly one size-2 bucket costs `1 + 2(n−3) + 5 = 2n` (exact singleton + `n−3` singletons + the pair). Easy to miscount; I re-derived it.
- **The sweep's `guess_lb`** is exactly `count + Σ lb(bucket)` — the `2−(c==1)` marginal telescopes to `2c−1 = lb(c)`, and `−counts[EXACT_MATCH]` cancels the exact bucket's `+1`. Both the unrolled ≤8 branch and the 243-piecewise branch agree.
- **Fail-soft semantics** at the `ub1==lb1` exact resolution and the tier-3 `bucket_beta` arithmetic are consistent with the "exact iff strictly under beta" invariant, which is what makes `evaluate_opener_*`'s `is_exact` flags honest.
- **The 128-bit Zobrist + "publish `size` last" ordering** in `SharedTT` is a correct lock-free protocol; the disjoint-union bound propagation (XOR of two disjoint subset hashes) is the coolest idea in the file.

## Bug found & fixed

**Dead greedy-aspiration seed** (`wordle_gemini.c:1752`): `main()` computed `initial_seed` for the top opener, printed it, then threw it away — `global_best_cost` was initialized to `UINT32_MAX`. In `--top`/`--all`, the *first* opener therefore ran completely unbounded (no ceiling), which is exactly the case where a greedy upper bound pays off most. Fixed to seed the pool, plus a reporting fallback so a seed that is never beaten (e.g. it's already optimal) is still reported as the incumbent instead of `UINT32_MAX` garbage. `--all` on the oracle fixtures and the 12-target fixture reproduce the Python reference exactly.

## The three optimizations (measured, not conjectured)

Profiling the exhaustive run: ~80% of time is the big-count candidate sweep (`wordle_gemini.c:770-804`), ~7% sorting, ~5% greedy seeding.

**1. Cool algorithm — monotone-LB early exit in the sweep** (`:817`). `guess_lb` is monotone non-decreasing in the per-guess pass, so the moment it crosses `beta` the guess can never beat the window — abort it and emit a `UINT64_MAX` sentinel key. Two subtleties that made this hard: (a) the `EXACT_MATCH` bucket contributes a `+1` marginal that's subtracted at the end, so an in-flight guess can still finish *one below* its in-flight lb — the abort threshold must be `beta + (exact_seen ? 1 : 0)`; (b) it's only worth the per-element compare when `beta ≤ 3·count`, so a `can_abort` gate keeps the branch free under huge beta. I hit bug (a) in practice (11494 ≠ 11483) and the fix restored correctness.

**2. Cool data structure — partial top-k selection replaces full sort** (`sort64_asc_top`, `:452`). Only the top `limit` candidates are ever visited, but the code quicksorted all 14,855 packed keys every node. `sort64_asc_top` is a quickselect (same median-of-3 Hoare partition as the existing sort) that converges the window containing the k-th smallest to ≤16 elements, then sorts `[0, lo)` and `[lo, hi)` — O(n) expected instead of O(n log n). Validated against `qsort` on 200k randomized trials.

**3. Low-level — a negative result that matters**: I tried an "active-list single-pass reset" (track touched scores instead of re-scanning `cols[i][g]`). It was *slower* in both modes (default 1.19s, exhaustive 26.3s!) — the data-dependent `c==0` branch in the hot loop costs more than the clean streaming reset it replaces. Reverted; that's what the empirical process is for. I also killed a `c<=243` comparison from a wrong per-*count* instead of per-*bucket* assumption that silently over-estimated `guess_lb` and broke the answer.

**Measured results (identical answers, 11483 everywhere):**

| Mode | Before | After | Δ |
|---|---|---|---|
| `--opener taler` default (10T) | 1.19s / 6789 n | 0.99s / 7015 n | **−17%** |
| `--opener taler --exhaustive` (1T) | 20.42s / 34,177 n | 18.03s / 35,687 n | **−12%** |
| `--opener taler --exhaustive` (10T) | 8.44s | 7.40s | **−12%** |
| `--top 10` (10T) | 3.55s | 3.25s | **−9%** |

## Lower-priority issues (left as-is, for your awareness)

- **TT memory**: 10 thread-local L1 TTs (2¹⁹ × 32B ≈ 16MB each) + 128MB shared L2. Packing `TTEntry` to 16B would roughly halve the footprint and improve probe hit latency.
- **`solver_tt_find`** (`:473`) copies L2's `proven_lower_bound` over L1's, potentially *weakening* a stronger local bound; perf-only, never unsound.
- **Unchecked `pthread_create`** (`:223`, `:1183`): a failed spawn is UB via `pthread_join`.
- **Tier-2 bucket hashing** (`:916`) is `O(count × buckets)`; it can be fused into the existing histogram pass for `O(count)`. The profile shows it's not currently a bottleneck, so I didn't pay the extra per-candidate XOR cost.

## Best next algorithmic lever

The sweep remains `O(num_guesses × count)` per node and is the residual 80%. The cleanest path to another ~2x is **vectorizing the sweep across targets** using a `[score][guess]` histogram laid out so consecutive guesses are contiguous, streaming the transposed matrix in blocks that fit L1 — i.e. trading the current scalar scatter-load inner loop for blocked, prefetch-friendly accumulation. It's a bigger surgery than the three fixes above, so I stopped at the measured wins.
