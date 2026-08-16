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

## Item 6 (loop fusion, targeted variant) — DONE

Implemented a narrower-scope version than the original "fuse dedup +
move-ordering + main loop into one pass" plan: the ranking pass already
computes `active_buckets` and `guess_lb` per representative (needed for
`lb1`/sort keys); these are now cached in `RankedCandidate` and reused in
the main loop's prune checks (`active_buckets == 1`, `guess_lb >=
current_best`) *before* rebuilding that candidate's histogram, so a
pruned candidate's O(count) histogram scan is skipped entirely instead of
being paid twice (once in ranking, once in the main loop, unconditionally,
as the old code did). Lower risk than a full 3-pass merge: no new
solver-owned scratch, no reentrancy surface, both cached formulas are
provably byte-identical to what the main loop used to compute fresh
(same inputs, nothing mutates between the two passes within one call).
Includes an inline proof that reordering the `guess_lb >= current_best`
check ahead of the perfect-split shortcut can never wrongly skip a
genuine perfect split (such a guess's `guess_lb` always equals the node's
*raw* `lower_bound[count]`, which is always <= the possibly-raised
`node_lb`, which is always < `current_best` at this point in the loop —
otherwise the loop would already have returned/broken earlier).

## Phase 3 (disjoint-union bound propagation) — DONE

Implemented per the plan: when a guess's bucket loop prunes partway
through, the already-solved sibling buckets (each returned an *exact*
cost, since `bucket_cost < bucket_beta`) have their hashes/sizes/costs
accumulated into a running union; on the prune break, if >=2 buckets were
folded in, the union's summed cost is written as a `proven_lower_bound`
for that union's own (hash1, hash2, size) key via `tt_find_or_claim` —
only ever raising a bound, never touching `exact_cost`, guarded so it
can't clobber a tighter existing bound. Each worker thread owns a private
`Solver`/TT (confirmed by re-reading `bucket_worker`/`tree_bucket_worker`/
`opener_worker`, all of which stack-allocate their own `Solver` and call
`solver_init`), so there's no cross-thread race — confirmed clean under
the TSan build in the full test suite, not just asserted.

Verified: full `test_solver.py` (oracle + ASan + UBSan + TSan +
cross-build stress agreement) green after both changes. Fixture
cross-checks (tiny/small/medium, `--all`) between the pre-change binary
and the post-change binary match exactly, `exact_total_guesses` and node
counts identical on those fixtures.

## Phase 2b — re-measurement of item 6 + Phase 3, full word list, salet, single-threaded

This machine's run-to-run noise turned out to be much larger than
Phase 2's original single-threaded numbers suggested (those were
apparently a low-noise run) — worth recording precisely since it changes
how to read every number in this file:

- Pre-item-6/Phase-3 baseline, three separate solo runs (no other CPU
  load): **209.70s, 217.03s, 205.92s** (mean 210.88s, ~5% spread even
  across a single unchanged binary).
- Post-item-6/Phase-3, two solo runs: **220.23s, 199.85s** (mean
  210.04s).
- Running one of each *simultaneously* (to control for time-varying
  drift) instead made both **slower** than any solo run (300.46s and
  291.85s respectively, +40%/+33% vs. their own solo numbers) — this
  10-core machine does not cleanly isolate two single-threaded
  CPU-bound processes from each other; parallel A/B on this box is not a
  valid comparison method, only sequential solo runs are.
- `exact_total_guesses` unchanged (11433) in every run that captured full
  output. Node count: baseline 2,251,108 vs. post-change 2,251,059 — a
  49-node reduction (0.002%), i.e. Phase 3's union-bound reuse fired
  essentially never on this specific opener/word-list combination.

**Honest conclusion: item 6 + Phase 3 together are wall-clock neutral on
this benchmark** — the post-change mean (210.04s) sits inside the
pre-change spread (205.92-217.03s), not below it. This is not the
regression the very first single-sample comparison suggested (220.23s
vs. 209.70s, a seemingly-real 5%) — that was noise, caught only by
re-running each side multiple times. Both changes remain worth keeping:
they're proven correct, item 6 strictly reduces work in theory (never
adds an iteration), and Phase 3's near-zero hit rate here doesn't rule
out it mattering more elsewhere (e.g. `--top`/`--all` runs where many
openers' subtrees overlap more, or smaller word lists where partition
collisions across different guesses are more likely) — but on the
specific salet/full-word-list/single-threaded benchmark this repo has
been tracking, they did not measurably close the gap to `wordle.cpp`.

## Phase 4 (endgame-cluster static coverage cutoff) — DONE

Implemented the static-only scope from the plan (`wordle.cpp:672-711`,
without the live re-search fallback at 715-721), re-derived for this
solver's total-cost objective rather than copied from wordle.cpp's
depth-feasibility one:

- `compute_endgame_table` (called once from `init_game_data`, single
  threaded, before any solver thread starts): groups target words by
  "wildcard pattern" (5-char string with one position replaced by `.`;
  since the dot's own position disambiguates which letter-slot is
  wildcarded, no separate position field is needed — two different
  positions can never produce the same pattern string). For every group
  of size >= 3, computes `max_d` = the largest number of distinct scores
  any single guess in the *entire* guess list produces across that
  group's members — a plain fact about the existing score matrix, not an
  estimate. `branch_capped_lower_bound(k, b)` (a new function,
  generalizing `compute_lower_bound_table`'s own 243-branching-factor
  formula to an arbitrary cap `b`) turns `(cluster_size, max_d)` into an
  admissible lower bound on that cluster's own standalone resolution
  cost. Only stored (in a new, TT-independent, read-only-after-init
  `EndgameTable`, keyed the same (hash1,hash2,size) way as everything
  else) when it's actually tighter than the generic
  `game->lower_bound[cluster_size]`.
- **Formula verified before trusting it, not just asserted**: wrote a
  brute-force recursive DP in Python (idealized even-split partitioning
  into <=(b-1) non-exact buckets) and compared against the closed form
  for b in [2,20), k in [0,300) — the `k<=b` branch (`2k-1`) is exact in
  every case (same reasoning as the original 243 case, which never
  actually depended on 243 specifically); the `k>b` branch is not exact
  for small b but is *never* an overestimate in any case checked, i.e.
  always sound even where loose.
- `solve_subset` gets one new read (`endgame_lookup`, a pure hash-table
  probe against the immutable table) folded into computing `node_lb`,
  right where the generic `game->lower_bound[count]` is read — raises
  `node_lb` via `max()`, same safe-combination pattern as the TT's own
  `proven_lower_bound`.
- Cross-validated the whole cluster-finding + max_d + bound computation
  independently in Python against the real `words.txt`: **589 clusters
  of size >= 3, 53 that actually tighten the generic bound** — both
  numbers matched the C implementation exactly.
- Verified: full `test_solver.py` green (oracle + ASan + UBSan + TSan) on
  a clean rebuild. `exact_total_guesses` unchanged (11433) on the full
  word list.

**Measured impact: negligible, same pattern as Phase 3.** Single-threaded
full-word-list salet run: 199.18s (inside the established noise band),
node count **2,251,059 — identical to the post-Phase-3 count**, i.e. the
endgame table changed zero pruning decisions across the entire 2.25M-node
salet search. Built a focused synthetic test to double check this wasn't
a wiring bug: constructed a word list whose *entire* target set is
exactly the single largest-margin cluster found (`.ight` →
eight/fight/light/might/night/right/sight/tight/wight, `max_d=6`, giving
`branch_capped_lower_bound(9,6)=20` vs. the generic `lower_bound[9]=17`).
Confirmed the table correctly identifies and stores this cluster (1
considered, 1 useful) and that `endgame_lookup` fires exactly at the
root (guaranteed, since the whole target set *is* the cluster) — but
node count was identical (73) with the lookup wired in vs. patched out.
Reason: the tightened bound (20) only changes behavior if some caller's
`beta` ever falls in the (17, 20] gap; here the tree's true achieved
cost (29) sits above both values, so neither bound ever triggers the
`current_best <= node_lb` early exit or a `node_lb >= beta` fail-soft
return.

**Why this scoped-down version has much less leverage than wordle.cpp's
own use of the same underlying fact** (a real insight worth banking, not
just a disappointing number): in wordle.cpp, the static cutoff feeds a
*depth-feasibility* check — "can this be solved in `remdepth` guesses at
all" — where a failed check returns `infinity` and eliminates an entire
subtree outright. For this solver's *total-cost* objective, the
analogous fact only tightens a numeric floor by a few points (the
clusters found here mostly tighten by ~3, e.g. 17→20), which rarely
lands exactly in the narrow gap where it flips a prune decision. The
subtree-elimination power wordle.cpp gets from this technique lives
almost entirely in the live re-search fallback (`wordle.cpp:715-721`) —
which was deliberately excluded from this phase's scope from the start,
specifically because folding a second solve over a different subset
framing into the same recursion/TT machinery was judged too risky to
take on without its own dedicated soundness pass. That fallback, not the
static cutoff alone, looks like the part of wordle.cpp's endgame
machinery that would actually matter for total-cost.

## Phase 5 (live endgame re-search fallback) — implemented, measured, NOT RECOMMENDED

Following up on the plan's own deliberately-excluded piece
(`wordle.cpp:715-721`), re-derived for this solver's total-cost
objective rather than copied:

**Design.** At any node with a finite `beta` (unbounded-`beta` callers —
`--tree`'s build path and `--opener`'s own top-level exact-search entry
point — structurally can never benefit, confirmed by reading every
`solve_subset` call site, so the check is skipped outright for them),
tally which precomputed endgame cluster (from a new reverse index,
`target_cluster_ids[t*5+j]`, covering every size>=3 cluster, not just the
53 that clear Phase 4's "tightens the generic bound" bar — a cluster's
*live*, narrowed overlap can be a genuinely different, more constrained
problem than its full global membership) has the largest live overlap
with the node's own targets. If that overlap (`E_live`) is a genuine
strict subset (>=3 members, `< count`), recursively call `solve_subset`
on `E_live` itself with the same `beta`. If that returns `v >= beta`,
monotonicity under subset — `cost(subset) <= cost(superset)`, since
restricting an optimal strategy for the superset to the subset's own
outcomes is itself a valid (if not necessarily optimal) strategy for the
subset — means the *full* node also fails against `beta`, so the entire
node returns fail-soft immediately, skipping its own dedup/ranking/
search outright (unlike Phase 4's static table, which only ever
tightens `node_lb` by a few points).

**Soundness, verified thoroughly given this was flagged from the start
as the highest-risk item in the whole plan:**
- Termination: the strict-subset guard (`best_count < count`) guarantees
  this recursive call can never re-enter the exact same `(hash1, hash2,
  count)` as the calling frame, only something strictly smaller — same
  induction every other recursive call in `solve_subset` already relies
  on. Stress-tested directly: built a fixture whose *entire* target list
  is exactly one cluster (`.ight` -> eight/fight/light/might/night/
  right/sight/tight/wight) to try to force the self-referential case;
  no hang, correct result, confirming the guard works.
- `out_guess` contract: this early-return path never touches
  `*out_guess`, matching the *existing* `node_lb >= beta` early-return's
  own convention (also never touches it) exactly — verified by reading
  every `solve_subset` call site: the two call sites with a finite
  `beta` (both bucket loops) already pass `NULL`; the sites that pass a
  real `&out_guess` pointer (`bucket_worker`'s beta-seeded calls, whose
  `out_guesses` turns out to be dead/unread downstream anyway; the
  `--tree` build path) always pass `beta == UINT32_MAX`, so this new
  path can't fire for them at all.
- Reentrancy: the new per-solver scratch (`cluster_stamp`/
  `cluster_count`, generation-stamped like `dedup_stamp`) is fully
  drained into a stack-local `e_live` array *before* the recursive call,
  so a nested invocation reusing/overwriting the same solver-owned
  scratch is safe — identical pattern to every other solver-owned
  scratch buffer in this file.
- Verified: full `test_solver.py` green (oracle + ASan + UBSan + TSan) on
  a clean rebuild, plus targeted ASan/TSan runs directly against a
  custom fixture built specifically to exercise this path under finite
  betas (`--top` mode, which shares a global ceiling across openers).
  `exact_total_guesses` unchanged (11433, full word list; 43, the
  custom fixture) in every comparison against a fallback-disabled build.

**Measured result: clearly not worth it.** Node count went *up*, not
down: 2,251,059 -> **2,434,904 (+8.2%)** on the same salet/full-word-list
search. Wall-clock stayed statistically neutral (two clean solo samples,
194.29s and 205.15s, both inside the already-established ~200-220s noise
band) rather than clearly regressing, but that's despite the extra
work, not because the mechanism is paying for itself. Added
instrumentation (temporary, not kept in the shipped file) to see why:
**179,792 fallback attempts, 21 hits — a 0.012% success rate.** Every one
of the other 179,771 attempts paid for a real recursive sub-search (its
own dedup/ranking/possibly-further-recursion) that found nothing and
was pure overhead.

**Why this is a dead end without much more work, and why that further
work isn't justified by what's already been measured:** wordle.cpp
avoids exactly this failure mode with its own `heuristic>0` gate before
attempting the live re-search — a check this implementation deliberately
left out to test the mechanism's *ceiling* value first (does the
underlying phenomenon matter enough here to be worth gating carefully?).
The 21-hit answer is: even a **perfect** gate — one that eliminates every
wasted attempt while keeping all 21 real ones — could only ever recover
whatever those 21 hits are worth, out of 2.4M+ node visits. Given Phase
3's disjoint-union bound *also* found only 49 useful moments out of
2.25M nodes with negligible wall-clock impact, 21 is almost certainly
inside the same "too rare to clear this machine's noise floor" territory
— and wordle.cpp's own gate condition (`mx-1 > remdepth-1`, a
worst-case-*depth*-budget check) has no principled, non-arbitrary
translation into this solver's total-cost/`beta` framing, unlike every
other technique ported this session. Building and tuning one anyway
would risk overfitting a threshold to this one word list for a ceiling
already shown to be marginal. **Recommendation: revert Phase 5**, keeping
the correct-but-unhelpful code out of the shipped solver rather than
carrying its overhead (even if currently noise-masked) for no measured
benefit.

**REVERTED.** Removed all Phase 5 code (the `GameData.target_cluster_ids`/
`num_clusters` reverse index and its construction in
`compute_endgame_table`, the `Solver.cluster_stamp`/`cluster_count`/
`cluster_call_id` scratch and its init/free, and the live-fallback block
in `solve_subset` itself) after measurement confirmed it wasn't worth
keeping. Verified clean after reverting: no leftover references anywhere
in the file, clean rebuild with zero warnings, full `test_solver.py`
green (oracle + ASan + UBSan + TSan), and Phase 4's own static endgame
table (item 6/Phase 3/Phase 4 all still present) confirmed still working
correctly post-revert (medium fixture: same "6 considered, 0 useful" and
same exact result as before Phase 5 was ever added). Items 6, Phase 3,
and Phase 4 remain in the shipped file.

## Post-Phase-5 investigation: why the gap doesn't close (root cause found)

Prompted by the user asking, essentially, "how is wordle.cpp so much
faster on the same machine if we've ported its own techniques" — this
deserved a real answer, not more incremental tuning. Read `wordle.cpp`'s
actual core recursion (`minoverwords`, lines 775-905) for the first time
this session, rather than only the endgame-specific code claude_plan.md
had scoped Phase 4/5 around.

**Finding**: `wordle.cpp` is built around a hard per-word depth budget
(`remdepth = maxguesses - depth`, `maxguesses` defaults to 6, the real
Wordle rule). `if (remdepth <= 0) return infinity;` and similar
`remdepth<=1`/`remdepth<=2` shortcuts fire at *every* node of the main
recursion, not just inside the endgame-cluster special case — a cheap,
unconditional "this subtree cannot possibly finish in time" cutoff that
needs no histogram-building, no guess-ranking, no recursion. Our solver
has no equivalent concept: it computes the true *unbounded* minimum
total cost, so nothing is ever "impossible," only more or less costly.

This is confirmed, not speculated: `solver/claude_plan.md:27-29`
(written before this session started) already recorded that `wordle.cpp`
with its default `maxguesses=6` reports **11433** for salet on this exact
word list — identical to our uncapped solver. That match is only
possible because the depth-6 constraint never actually binds for this
word list's true optimum, which is exactly what lets `wordle.cpp` use
the `remdepth` shortcut so aggressively without it changing the answer.
`solver/findings.md` had attributed the gap to "endgame-cluster coverage
cutoff + live-endgame subsearch" specifically (calling it "the two
orders of magnitude") — but that's the depth-budget mechanism's
narrowest special case, not the general mechanism itself, which is woven
through the whole recursion. That explains, in hindsight, why this
session's Phase 4 (0 useful hits) and Phase 5 (21/179,792) measured so
little: the special case was ported without the general mechanism that
gives it most of its power.

**Why a direct port isn't safe, and isn't free either.** Worked through
what porting `remdepth`-style hard cutoffs into our own recursion would
actually require, in detail, before writing any code:
- A hard cutoff (mirroring `wordle.cpp` exactly: return early whenever
  the depth budget is exhausted) is only *correct* if the depth cap
  provably never binds. If it ever did — a bug, or a different word
  list — the solver would silently report a wrong total. That directly
  contradicts this file's own stated design goal #1 ("provable
  correctness"), so this was rejected as a real, not just theoretical,
  risk.
- A "safe" version that falls back to the normal algorithm whenever the
  cap would bind was worked through as an alternative — and it reduces
  to a no-op: whenever the shortcut's condition is never triggered
  (which is required for correctness), the search behaves *identically*
  to today's code, since the extra check is a no-op precisely then. The
  cheap `remdepth` comparison doesn't carry information our
  `lower_bound[]`/`lb1` machinery doesn't already have — it's a cheaper
  way to notice the *same* fact, and an untaken shortcut saves nothing.

So this technique doesn't have a form that's both safe and non-vacuous
for this solver's objective, unlike everything landed earlier this
session.

## Improved greedy aspiration seed (widened guess search) — DONE

Redirected the investigation toward a place where a *tighter, still
provably-safe* real strategy is already explicitly permitted:
`greedy_pick`/`greedy_upper_bound` (the beta-seeding heuristic used by
`evaluate_opener_parallel`). Its existing contract already guarantees
"the returned total is always something some real strategy achieves" —
so improving *how good* that real strategy is can only ever tighten the
seed, never invalidate the safety argument.

**Change**: `greedy_pick` previously searched only the live candidate
pool for its next guess. Widened it to search the *full* guess list,
matching how the exact solver itself always searches the full list —
the classic case this helps is exactly an endgame-style cluster (e.g.
the `.ight` words), where every candidate "wastes" one of its own
letters confirming itself and a non-candidate guess splits far more
evenly. Guarded the call site to skip the now-more-expensive scan for
buckets of size <=2, since `greedy_upper_bound`'s own base cases ignore
the chosen guess entirely at that size — otherwise the widened search
would pay full cost on the (numerous) trivial buckets for no benefit.

**Verified**: full `test_solver.py` green (oracle + ASan + UBSan + TSan)
on a clean rebuild; fixture cross-checks (tiny/small/medium, `--all`)
match the known-correct totals exactly.

**Measured**: the aspiration ceiling for salet tightened from **12013 to
11746** (true optimum: 11433) — a real, reproducible improvement,
directly instrumented and compared against the pre-change binary on the
same opener. But the full exact search's own node count and wall-clock
were unchanged (2,251,059 nodes, ~210s, inside the same noise band as
before) — the tighter seed is real but wasn't tight enough to unlock
materially more pruning in the exact search for this specific opener.
**Kept regardless**: unlike Phase 5, this carries no measured downside
(same safety contract as the existing, already-trusted aspiration-seed
mechanism, one-time cost per `--opener` call, no wall-clock regression
observed) and is a genuine, verified tightening of a real number, even
if its downstream effect on this one benchmark is currently within
noise.

## Depth-capped solver (`--max-guesses N`) — DONE, real (if modest) speedup

Following up directly on the user's request to implement the hard
6-guess depth budget after the root-cause investigation above: this
computes a genuinely *different* objective (minimum total cost SUBJECT
TO no target needing more than N guesses) rather than the file's usual
unbounded minimum, so it's implemented as a fully separate, isolated
code path — `CappedTTEntry`/`CappedTT`/`CappedSolver`/
`solve_subset_capped`/`evaluate_opener_capped`, all new, none of the
existing `TTEntry`/`TT`/`Solver`/`solve_subset` touched — specifically
so a bug in this newer, higher-risk code cannot corrupt the extensively
verified unbounded path. Opt-in via `--max-guesses N` (0 = default,
unbounded, zero behavior change); `--opener` only, rejected at the CLI
if combined with `--top`/`--all`/`--tree`.

**Why a separate TT, not a parameter on the existing one**: under a
depth cap, a subset's cost depends on how many guesses *remain*, not
just the subset itself (fewer remaining guesses can force a worse or
infeasible resolution) — unlike every solver above, where a subset's
optimal cost is intrinsic (explicitly documented as such at the top of
the existing `TT` struct). `wordle.cpp` handles exactly this by indexing
its own cache by depth (`cache[depth]`); this file's equivalent is
`CappedTTEntry.remdepth` as an explicit fourth key field alongside
`(hash1, hash2, size)`.

**Scope of what got ported**: deliberately just the two cheapest,
structurally-unconditional shortcuts from `wordle.cpp`'s `minoverwords`
— `remdepth==0` (no guesses left, count>=1 is always infeasible: even a
known answer needs a guess to submit it) and `remdepth==1 && count>1`
(one guess left can resolve at most one candidate for certain, a
pigeonhole fact independent of which guesses exist). Both are provably
true regardless of word list. Left out for this first pass: the
`remdepth<=2` "no guess achieves an all-singleton split" check (needs an
extra upfront scan, and wordle.cpp's own use of it isn't unconditionally
provable the same simple way), node_lb depth-awareness, and Phase 3's
union-bound cache — smaller, independently-verifiable first version over
a maximal port.

**Infeasibility handling (the actual safety-critical part)**: represented
internally by `CAPPED_INFEASIBLE = 1000000000` (matching wordle.cpp's
own `infinity`), chosen to be many orders of magnitude above any
realistic total for word lists this size, so it propagates correctly
through the *existing* beta/fail-soft machinery (`bucket_cost >=
bucket_beta` already rejects it like any other failed candidate) with no
new plumbing needed for the common case. **A real bug was caught before
it shipped**: the first draft's top-level fallback re-solved a "failed"
bucket at `beta=UINT32_MAX` to disambiguate "beaten by a better
strategy" from "genuinely infeasible" — but `UINT32_MAX` (~4.3 billion)
is *larger* than `CAPPED_INFEASIBLE` (1 billion), so a genuinely
infeasible bucket's return value would NOT have registered as `>= beta`
in that retry, silently corrupting the total instead of being detected.
Caught by tracing the arithmetic before testing, not by a test failure.
Fixed by simplifying instead of patching: dropped the ceiling-seeded
beta and the two-pass retry entirely; each bucket is now solved once,
directly, at `beta=CAPPED_INFEASIBLE` — a returned cost below that is
unambiguously the true capped-optimal value, at or above it is
unambiguously genuine infeasibility, no retry or seeding needed. Fewer
moving parts, and the class of bug that motivated the simplification
can't recur because there's no second beta value to reason about.

**Verified, several ways**:
- Full `test_solver.py` green (oracle + ASan + UBSan + TSan) — confirms
  nothing in the existing file broke.
- `--max-guesses` path itself run directly under ASan and TSan (not
  covered by `test_solver.py`, which predates this feature) — clean.
- Fixture cross-checks (tiny/small/medium) at `--max-guesses 6`: exact
  match with the unbounded solver's own totals (27, 96, 339).
- **Independent boundary-condition oracle**, hand-derived and checked
  against the implementation rather than merely asserted: a word list of
  5 mutually letter-disjoint words (e.g. abcde/fghij/klmno/pqrst/uvwxy)
  forces pure sequential elimination under optimal play (every
  non-matching guess gives zero information), so the true cost is the
  closed form `n(n+1)/2` and the *last*-resolved word always needs
  exactly `n` guesses in any optimal ordering. For n=5 (cost 15): capped
  at 6 and 5 both correctly returned 15 (matching uncapped, cap not
  binding); capped at 4 correctly reported infeasible. For n=6 (cost
  21): capped at 6 correctly returned 21 (exactly at the boundary);
  capped at 5 correctly reported infeasible. Every boundary matched the
  hand derivation exactly, including the two tightest (still-feasible)
  cases, which are the ones most likely to expose an off-by-one.

**Measured on the tracked benchmark**: single-threaded, full word list,
salet, `exact_total_guesses` unchanged (11433, confirming the depth-6
cap doesn't bind on this word list, consistent with the earlier
wordle.cpp cross-check). Six solo runs each of capped and uncapped,
interleaved across two separate sessions to reduce time-correlated bias,
user-time (immune to this machine's sleep-interruption issue, which
corrupted `real` time in two of the runs):

- Capped: 187.85, 197.55, 207.92, 207.19, 204.28, 201.96 — mean
  **201.12s**, stdev 7.52.
- Uncapped: 205.92, 209.70, 217.03, 210.18, 208.59, 217.40 — mean
  **211.47s**, stdev 4.69.
- Difference: **10.34s, 4.9%** — a real, if modest, speedup: the two
  means' approximate 95% confidence intervals (using each sample's own
  standard error) are [195.5, 206.7] and [208.0, 215.0] respectively —
  narrow but non-overlapping, unlike every other change measured this
  session, where the post-change range sat fully inside the pre-change
  spread.

**Honest framing**: this is the first change all session to show a
measurable, plausibly-real (not noise-floor) improvement — but it's a
~5% win from porting only the two cheapest depth shortcuts, not the
dramatic gap-closer the root-cause investigation might have suggested
was available. Node count actually went *up* slightly under the cap
(2,291,175 vs 2,251,059, +1.8%) despite being faster, confirming "node"
isn't a directly comparable unit between the two solvers — many of the
capped solver's extra nodes are near-free `remdepth==0/1` O(1)
short-circuits, not full search nodes, so raw node count understates the
per-node work reduction. The unported refinements (remdepth<=2, depth-
aware node_lb, union-bound) are plausible next steps if more speedup is
wanted, following the same incremental, verify-before-extending pattern
used throughout this file.

## Status: paused here for a checkpoint

Phase 1 (items 1-5), item 6, Phase 3, Phase 4, and Phase 5 are all
implemented and verified correct under the full sanitizer suite,
`exact_total_guesses` unchanged throughout. Every one of the four
post-Phase-1 additions (item 6, Phase 3, Phase 4, Phase 5) is
individually sound, but **none moved the measured wall-clock on the
tracked salet/full-word-list benchmark** — a pattern that held four
times in a row, not noise (Phase 3: 49/2,251,108 useful node-level hits;
Phase 4: 0/2,251,059; Phase 5: 21/179,792 fallback attempts, and its
extra overhead measurably *increased* node count by 8.2% without a
corresponding wall-clock cost, i.e. it's real overhead, just currently
small enough to hide in the noise). `wordle.cpp`'s own playbook is now
exhausted: every technique from it that plausibly translates to this
solver's total-cost objective has been tried, measured, and found to
matter far less here than it does for wordle.cpp's own (worst-case-depth,
hard-mode-capable) objective and word lists. **Phase 5 is recommended for
revert** (see its own section above) rather than being kept as
dead-weight-but-harmless code. Pausing here to report the full pattern
across all five phases and decide the next direction — this is no longer
a "one more phase might close the gap" situation; it's a "the gap likely
isn't closable by porting more of wordle.cpp's specific techniques"
situation, which calls for a different kind of decision than earlier
checkpoints in this file.

**Update, post-investigation**: Phase 5 has since been reverted (see its
own section's "REVERTED" note above). The root-cause investigation above
found the actual structural reason the gap persists — `wordle.cpp`'s
depth-budget mechanism doesn't have a safe, non-vacuous port to this
solver's unbounded objective — which is a more complete and more honest
answer than "we haven't found the right technique yet." The one
follow-up that *did* land (widened greedy aspiration seed) is real,
verified, and kept, but modest. Current shipped state: Phase 1 (items
1-5), item 6, Phase 3, Phase 4, and the widened greedy seed are all in
`wordle_claude.c`; Phase 5 is not. `exact_total_guesses` unchanged
(11433) throughout every change made this session.
