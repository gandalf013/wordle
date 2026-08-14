# Speedup techniques for the exact Wordle solver

Catalog of techniques for making `wordle_claude.c` (or whatever exact/optimal
solver comes next) fast enough to run to completion, plus a priority order.
Compiled from: a close read of `alex1770/wordle`'s `wordle.cpp` (Alex Selby's
solver, the one that found the "salet" 3.4212 optimum), a review of our own
`solver/wordle_claude.c` and `solver/bugs.md`, and research into other
Wordle-solver implementations, classical exact Mastermind algorithms, and
general game-tree/branch-and-bound search techniques.

No item here has been implemented yet. This is a planning document.

**2026-08-14 cleanup:** `bugs.md` (the source for some citations below) was
deleted after being folded into this file. Separately, `wordle_claude.c`'s
mid-recursion dynamic-budget-donation parallelism (`reserve_budget`,
`TeamPool`, `team_worker*`, `run_team_pool`, `solve_subset_team`,
`TEAM_TOP_K`) was removed as premature optimization: it wasn't needed by
any Tier 1-3 item, no profiling had shown it load-bearing, and it added
real complexity (atomics, mutexes, thread spawn at arbitrary recursion
depth) to exactly the function (`solve_subset`) that Tier 1/2 heuristics
need to be added into next. Parallelism is now flat and single-level only:
across root buckets of one opener, or across openers. See items 21-22.

## A. Node-level pruning / lower bounds

1. **Admissible size-based lower bound** (`2k-1` for small buckets,
   `3k-1-buckets` generally) — have, in both `wordle.cpp` and
   `wordle_claude.c`.
2. **Endgame-cluster / wildcard-coverage cutoff** — static bound on whether
   remaining guesses can possibly distinguish a "one-letter-off" cluster of
   hidden words (e.g. `co.ed` matching `coked/comed/...`); falls back to
   searching just the smaller "live endgame" set when not statically cut.
   Missing. (wordle.cpp)
3. **Indirect/derived-bound propagation to sibling nodes** on a beta
   cutoff — merges score-buckets across one or two letter positions to
   cheaply populate bounds for *other* nodes than the one just computed.
   Missing. (wordle.cpp)
4. **Information-theoretic entropy bound**: Ω(log(candidates)/log(branching
   factor)) guesses needed — cheaper, more general admissible bound than the
   combinatorial one; usable as a first-pass check before the tighter
   combinatorial bound. **Skipped, not just deferred:** our combinatorial
   `lower_bound` table is already an O(1) lookup (precomputed closed form,
   not a recursive/expensive computation), and is strictly tighter than
   this bound in every case that matters here. A weaker O(1) bound gains
   nothing when the tighter bound is equally cheap to consult -- there's no
   "expensive tight bound, cheap loose pre-filter" tradeoff to exploit in
   this codebase the way there might be elsewhere. Revisit only if the
   combinatorial table ever stops being O(1) (e.g. item 25's bitmask work
   or a future non-closed-form bound).
5. **Dominance/subsumption pruning between candidate guesses** — if guess
   A's partition of the current subset strictly refines guess B's, B can be
   dropped outright, not merely deduped when identical (current code only
   dedups exact-identical partitions). Missing, generalizes existing dedup.
   Still deferred with 2/3 (see Tier 2) -- the refinement check itself is
   straightforward (compare each candidate's per-target scores against the
   TT-suggested/best-so-far guess's, using the score matrix already in
   hand), but it's real new surface area and the priority notes' own
   ordering ties its value to endgame clusters, which don't exist yet.
6. **Fast single-pass perfect/near-perfect-split shortcuts** — **have**
   (2026-08-14): `solve_subset` now detects, from the histogram alone
   (`active_buckets == count && hist[EXACT_MATCH] > 0`), a guess that
   exact-matches one live candidate and splits every other one into its
   own singleton bucket. That guess's cost is exactly `lower_bound[count]`
   -- not an estimate, since every singleton bucket's cost (1) is a base
   case, not a recursive lower bound -- so it's unconditionally optimal
   and the search stops immediately, without sorting/partitioning buckets
   or recursing into any of them. Only the exact-match ("perfect self
   split") case is handled; a perfect split by a guess *not* among the
   live candidates is detectably exact too (no recursion needed) but isn't
   proven globally optimal the same cheap way, so it's left to the normal
   candidate loop -- lower expected value, not implemented.
7. **Explicit closed-form base cases for tiny subsets** — **have**
   (2026-08-14): `count==2` is now a direct base case (mirrors `count==1`),
   returning `lower_bound[2] = 3` -- the only achievable outcome shape for
   2 distinct candidates (guess one, free exact-match or a forced size-1
   remainder), so no guess search is needed. Verified against
   `reference_solver.py`'s independent brute-force oracle.
8. **Reversible-move / Conway pruning** — **have** (2026-08-14): a
   candidate guess with `active_buckets == 1` (every remaining target
   scores identically) is skipped outright. It can never be optimal --
   count >= 3 at this point (count <= 2 are base cases), and some other
   candidate is always guaranteed to make real progress (any live target,
   guessed, resolves itself for free) -- and, more sharply, recursing into
   it would re-enter the *same* (hash, count) subset from within its own
   still-executing call, which the TT can't shortcut since it only caches
   completed results. This was a latent non-termination/stack-overflow
   risk on any word list with a genuinely zero-information guess in the
   candidate pool, not just a speed gap.

All three (6, 7, 8) were checked against `reference_solver.py`'s
independent oracle (`test_solver.py`'s `ORACLE_CASES`), under plain, ASan,
and TSan builds, plus cross-build agreement on the larger stress fixtures
-- see `test_solver.py`. Fixture-scale node-count reduction measured via
`benchmark_solver.py --compare`: -52% (tiny), -35% (small), -33% (medium),
with `exact_total_guesses` unchanged in every case.

## B. Move ordering (which guess to try first, to tighten beta fast)

9. **sum_sq / active-buckets heuristic ranking** — have (both codebases,
   converging design).
10. **Killer-move heuristic** — track, per remaining-depth (not per exact
    subset), the guess that most recently caused a cutoff there, try it
    first on siblings. Missing. (chess engines)
11. **History heuristic** — global (not per-depth) score per guess word,
    incremented on every cutoff anywhere in the tree, used as an ordering
    tie-break. Would let the solver "discover" generically strong words
    over a run. Missing. (chess engines)
12. **TT-suggested-guess reuse** — have (`suggested_guess` swapped to front
    of ranked list). Killer/history above are complementary: they fire even
    when the TT has no entry for this *exact* subset.
13. **Bucket-processing order within a single guess's partition** — our
    code sorts buckets largest-first (`compare_bucket_size_desc`); Tseng's
    writeup specifically credits smallest-bucket-first processing as one of
    the two biggest wins in his implementation, for reaching cutoffs
    faster. Direct, checkable disagreement with our current code.
    (sonorouschocolate.com)
14. **Knuth-style greedy minimax guess selection** as the ranking heuristic
    (minimize *worst*-bucket size) — different objective from our
    sum_sq/entropy-ish ranking; could be tried as an alternate/blended
    signal. (Mastermind literature)

## C. Caching / memoization

15. **In-memory TT keyed by (subset hash, size), generation-independent** —
    have, and correctly avoids the generation bug found in
    `wordle_solver.c` (see `bugs.md` #3).
16. **Disk-persisted, path-independent cache** (keyed by
    `(testwords, hiddenwords)`, survives across runs/openers) — missing.
    Turns a killed/timed-out run into salvageable progress instead of a
    total loss. (wordle.cpp)
17. **Bottom-up small-subset tablebase precomputation** — exhaustively
    precompute exact costs for every distinct small subset that actually
    occurs, before the main search. Initially looked promising as a fix for
    the O(num_guesses) per-node guess-list scan, but on reflection its
    value is mostly subsumed by the existing TT (for recurring subsets) and
    by #2/#3 (which avoid the scan on *new* awkward subsets by proving a
    cutoff without evaluating candidates individually). Demoted — see
    priority notes below.
18. **Global history-heuristic table** (item 11) doubles as a lightweight
    cross-run cache of "generically good words," cheaper than the full disk
    cache in #16.

## D. Search framing / tree traversal

19. **Upper-bound seeding (aspiration window)** — run a cheap heuristic
    solve (entropy-greedy, minimax-greedy, or reuse the existing Python
    strategies) first to get a real, tight `beta` before the exhaustive
    pass, instead of starting from infinity. Every source frames this as
    *necessary groundwork*, not a marginal tweak — the exhaustive pass
    becomes verification, not from-scratch minimization. Missing.
    (wordle.cpp's `-b` flag exists for this; general aspiration-window
    practice)
20. **Iterative deepening / MTD(f) refinement** of that seed — re-tighten
    the window incrementally rather than one-shot. Lower confidence this
    helps much beyond a single good seed, since Wordle's cost objective
    isn't smooth like a chess eval. Missing, speculative value.
21. **Young Brothers Wait Concept** — fully search the best-ordered child
    sequentially first (to get a tight bound) before fanning remaining
    siblings to helper threads, rather than parallelizing blind. Missing
    (the mid-recursion dynamic-budget-donation scheme this item originally
    referred to as a base to refine was removed in the 2026-08-14 cleanup
    pass as premature optimization — see below; parallelism is now flat,
    root-bucket/root-opener only, so YBWC would be a fresh addition, not a
    refinement).

## E. Parallelism / systems

22. **Per-thread TT, flat root-level parallelism** — have. (Originally
    also had a dynamic, mid-recursion, atomic-budget-donation scheme that
    let helper threads nest in at any depth; removed in the 2026-08-14
    cleanup as complexity that wasn't serving any planned Tier 1-3 item
    and that no profiling had shown was load-bearing. Root-bucket
    parallelism (`bucket_worker`/`evaluate_opener_parallel`) and
    root-opener parallelism (`opener_worker`) remain and are simple,
    embarrassingly-parallel, and correctness-independent.)
23. **Lazy-SMP-style single shared (sharded/lock-free) TT** across all
    threads instead of per-thread TTs — lets any thread's discovery help
    any other thread immediately, not just within a donation event.
    Missing, real architectural alternative worth weighing once there's
    profiling data showing TT contention (or the lack of nested
    parallelism) actually costs something.
24. **Root splitting across openers** — have (`opener_worker` /
    `evaluate_opener_parallel`).

## F. Low-level / data representation

25. **Bitmask-based target subsets** instead of index arrays, for cheaper
    partition/histogram computation and hashing. Note: precomputing full
    per-guess-per-score bitmasks against the *entire* target list is not
    memory-feasible (~14k guesses × 243 scores × ~36 words ≈ far too
    large); the realistic version is representing just the *current node's*
    subset as a bitset for cheaper iteration/hashing — a modest constant
    factor, not a dramatic one. Missing.
26. **AVX2 vectorized histogram/popcount** — up to ~2x on the innermost hot
    loop (`hist[score]++` over all targets). Unclear/missing in the solver
    itself (the git history's "fused bincounting" work was for the Python
    scoring kernel, not confirmed present in `wordle_claude.c`).
27. **PEXT/PDEP (BMI2) bit tricks** — useful for sparse bit selection (e.g.
    green/yellow position masks); `wordle.cpp`'s `okhard` hard-mode filter
    already uses a related SWAR trick. Caveat: notoriously slow on AMD
    Zen 1/2, needs a runtime-dispatch fallback if portability matters.
    Missing.
28. **Cache-tiled/transposed score-matrix layout** for the
    `hist[sc[h][t]]++` inner loop, since `hwsubset` is a scattered index
    set into a large flat matrix. Missing.
29. **Zobrist hashing** — have, nothing new here.

## G. Structural / architectural alternatives

30. **Mixed-Integer Optimization formulation** (Bertsimas et al.) — solve
    the *entire* decision tree jointly via a MIP solver instead of
    recursive branch-and-bound. Fundamentally different paradigm; not a
    drop-in technique for a hand-rolled C solver, but useful as a
    sanity-check/verification method against whatever our exact solver
    eventually produces.
31. **Separate "build best tree" vs. "exhaustively verify optimality"
    passes** (Jonathan Olson) — treat the two as different problems with
    different budgets/techniques rather than one generic recursive
    function serving both; a serialized/persisted decision-tree artifact
    as the deliverable, not just a number. Related to #16.

## Sources consulted

- `alex1770/wordle`'s `wordle.cpp` (Alex Selby) — direct read of the source.
- poirrier.ca/notes/wordle-optimal, poirrier.ca/notes/wordle — Laurent
  Poirrier's aggregation of known results and hard-mode lower bounds.
- sonorouschocolate.com — Peter Tseng's independent implementation and
  notes on partition-processing order and word ordering.
- jonathanolson.net/experiments/optimal-wordle-solutions — Jonathan
  Olson's build/verify split and serialized decision-tree approach.
- Bertsimas et al., "An Exact and Interpretable Solution to Wordle" — MIP
  formulation.
- nkoppel/OptimalWordleSolver (Rust) — memory-footprint corroboration at
  full-dictionary scale.
- Knuth (1977), Irving (1978-79), Koyama & Lai (1994) — classical exact
  Mastermind minimax/expected-case algorithms.

## Priority order

Grouped into tiers. Two threads run through this ranking: (1) this project
has a documented history of pruning heuristics that silently drop
correctness (`bugs.md`), so anything that *skips* candidates without a
proof gets extra scrutiny; (2) cheap, low-risk, universally-endorsed wins
go first regardless of theoretical ceiling.

### Tier 1 — do these first (cheap, low-risk, high expected value)

1. **#19 Upper-bound seeding (aspiration beta).** `solve_subset`
   presumably starts from `beta = infinity` (or close to it) at the root.
   Every source treats a tight starting bound as necessary groundwork —
   it changes the first exhaustive pass from "discover a good answer" to
   "verify a good answer." Implementation is trivial: run an existing
   Python strategy once (or a quick one-ply greedy C pass) to get a real
   playable tree, feed its cost in as `beta`. Zero correctness risk — it
   only tightens pruning, never changes what's considered.
2. **#7 Explicit `count==2` base case.** One `if` statement, mirrors the
   existing `count==1` case, structurally forced (not a heuristic — there
   is only one sane strategy for 2 candidates). Count-2/3 nodes are an
   enormous fraction of all calls near the leaves.
3. **#13 Bucket-processing order experiment.** Flip
   `compare_bucket_size_desc` to smallest-first behind a compile-time flag
   and A/B it — Tseng credits this order as one of his two biggest wins.
   Free to try; disagreement with a known-good implementation is worth
   chasing down before inventing anything new.
4. **#10/#11 Killer-move + history heuristic.** Small fixed-size arrays
   (`killer[depth]`, `history[guess]`), updated on every cutoff, consulted
   before falling back to `sum_sq` ranking. Doesn't change which guesses
   are searched — pure move-ordering, as safe as the existing
   `suggested_guess`-from-TT swap already in place.

These four are independent of each other and of everything below — do them
together, benchmark, then move on.

### Tier 2 — the real algorithmic centerpiece (higher effort, highest ceiling, needs care)

5. **#2 Endgame-cluster cutoff** + **#3 Indirect bound propagation.** This
   is what gave Selby's solver its "two orders of magnitude." Both target
   the failure mode where the solver has no choice but to brute-force
   through thousands of probe-word candidates for an awkward one-letter-off
   cluster. This is also the real answer to the O(num_guesses)-per-node
   overhead: rather than making that scan cheaper, this avoids needing to
   run it at all on the nodes where it's most expensive, by proving a
   cutoff or shrinking to the live-endgame subset first. Correctness note:
   unlike most of the forward-pruning bugs in `bugs.md`, this technique is
   sound as alex1770 states it (a proven coverage bound, not a
   sampled/capped heuristic) — but it's the most intricate thing on this
   list to implement right, and this codebase has a track record of subtle
   bugs in exactly this kind of reasoning. Budget real review/testing time,
   not just implementation time. Do this only after Tier 1 is in and
   benchmarked, so its marginal effect is measured against an already-
   reasonable baseline.
6. **#5 Dominance/subsumption pruning**, scoped down: check refinement only
   against the current best-so-far and TT-suggested guess (cheap, catches
   the common case), not all-pairs (expensive). Pairs well with #2/#3,
   since endgame clusters are exactly where many guesses induce redundant,
   non-refining partitions.

### Tier 3 — worth doing once Tier 1+2 are in and profiling shows where time actually goes

7. **#16 Disk-persisted, path-independent cache.** Turns "run died at hour
   6" into resumable progress, and lets separate opener searches share
   subtrees. A resilience/throughput multiplier, not a speed fix — most
   valuable once a single run is expensive enough to want to checkpoint.
8. **#21 Young Brothers Wait Concept discipline.** Now a fresh addition
   rather than a refinement, since the dynamic-budget-donation scheme it
   would have refined was removed (see item 22's note). Would mean:
   search the best-ranked candidate at a node alone first, then fan out
   remaining ones. Best done after #10/#11 give a genuinely-good "first
   candidate" to search alone.
9. **#25 Bitmask-based subset representation**, scoped realistically (see
   note in item 25 above) — a modest constant-factor systems cleanup, not
   a priority item.

### Tier 4 — defer or drop

- **#17 Small-subset tablebase precomputation** — demoted; value mostly
  subsumed by the existing TT plus #2/#3. Revisit only if profiling after
  Tier 1+2 shows a specific remaining hot spot it would address.
- **#23 Lazy-SMP shared TT redesign** — high-effort, high-regression-risk
  rework of already-working, fairly sophisticated concurrency code. Only
  worth it if profiling shows TT contention is an actual bottleneck.
- **#26-28 SIMD/PEXT/cache-tiling** — classic last-mile optimizations.
  Premature before #2/#3 land, since there's no point hand-vectorizing a
  loop that better pruning will make mostly unnecessary. Revisit last.
- **#14 Knuth-style minimax ranking** — cheap experiment if curious, but no
  strong reason to expect it beats the existing sum_sq heuristic.
- **#20 MTD(f)/iterative deepening** — research was lukewarm on this
  transferring well to Wordle's non-smooth cost objective. Skip unless
  #19 alone proves insufficient.
- **#30 MIP formulation, #31 separate build/verify passes** — different
  paradigm / project-shape decisions, not modifications to the current
  solver. Worth keeping in mind as a validation method once there's
  something to validate, not as an implementation target now.

### Recommended concrete next step

Tier 1 (items 1-4) is small enough to do as one pass: seed beta, add the
count==2 case, flip the bucket-sort order behind a compile-time flag so it
can be A/B'd, add killer+history arrays. Benchmark against the current
baseline on a fixed subset of openers before touching anything in Tier 2,
so #2/#3's effect can be measured against an already-reasonable baseline
rather than conflated with the free wins.
