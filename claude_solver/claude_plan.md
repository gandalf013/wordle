# Plan: closing the speed gap with alex1770/wordle.cpp — claude_solver/

Supersedes `solver/claude_plan.md` for ongoing work in this directory
(that file is left untouched per instruction not to modify anything under
`solver/`). See `PROGRESS.md` in this directory for full detail on what's
been done and the measured numbers behind each line below.

## Status

**Phase 1 (items 1-5): done, verified.**
**Item 6 (loop fusion, targeted variant): done, verified.**
**Phase 3 (disjoint-union bound propagation): done, verified.**
**Phase 4 (endgame-cluster static coverage cutoff): done, verified.**
**Phase 5 (live endgame re-search fallback, the piece originally excluded
from Phase 4's scope): implemented, verified, measured, then REVERTED**
(kept out of the shipped file — see PROGRESS.md for the full removal
note and why).
**Phase 2/2b (re-measurement): done.**
Every item from the original plan, plus the one piece originally
excluded by deliberate scope decision, is now implemented and verified.
**None of items 6/Phase 3/Phase 4/Phase 5 measurably improved wall-clock
on the tracked benchmark** — see PROGRESS.md for the full, now
four-times-repeated pattern.

Headline result: single-threaded, full word list, salet opener —
864.4s (pre-Phase-1) → **~405s or ~210s (post-Phase-1)** — see PROGRESS.md's
Phase 2b note: this machine's true run-to-run noise (~5%, confirmed via
repeated solo runs) is large enough that the original single-sample
405.1s and this session's single-sample 209.7s cannot both be trusted as
precise; treat "roughly 200-400s on this machine, noisy" as the honest
Phase-1 number rather than either figure alone. Item 6 + Phase 3 + Phase
4 + Phase 5 on top of that: **wall-clock neutral across the board**, with
each phase's own diagnostic confirming *why*: Phase 3's union-bound reuse
fired on only 49 of 2,251,108 nodes; Phase 4's endgame table (589
clusters found, 53 that tighten the generic bound, cross-validated
independently in Python) changed zero pruning decisions on the same
search; Phase 5's live fallback fired 179,792 times but only ever helped
21 of them (0.012%), while its overhead measurably *increased* node count
by 8.2% (2,251,059 -> 2,434,904) without a matching wall-clock cost —
real, currently-noise-masked overhead, not a free option. Items 6/3/4
remain correct, cheap, and kept. **Phase 5 does not clear the same bar
and is recommended for revert** — see PROGRESS.md's Phase 5 section for
why a principled gate (the piece that would fix its overhead problem)
doesn't have a clean, non-arbitrary translation from wordle.cpp's
worst-case-depth framing into this solver's total-cost one, and why even
a hypothetically perfect gate could only ever be worth those same 21
hits. Gap to `wordle.cpp`'s 34.25s reference is still roughly ~6-10x
depending which baseline figure is trusted — every technique from
wordle.cpp's playbook that plausibly translates to this solver's
objective has now been tried and measured, so closing more of the gap
would mean a different kind of approach, not another port from
wordle.cpp.

## Scope (unchanged)

- Easy mode only. No item depends on `wordle.cpp`'s `oktestwords`-style
  hard-mode bookkeeping.
- Goal: make `--opener` and `--top`/`--all` usably fast on the real word
  list, not feature parity with `wordle.cpp`.

## Next steps, in order

### 1. Item 6 — inner-loop fusion — DONE (targeted variant)

Implemented as a narrower change than originally scoped: rather than
merging all three passes (dedup, ranking, main-loop histogram rebuild)
into one, the ranking pass's already-computed `active_buckets`/`guess_lb`
per representative are cached and reused by the main loop's prune checks
*before* that candidate's histogram is rebuilt, so a pruned candidate's
O(count) histogram scan happens once instead of twice. See PROGRESS.md
for the correctness argument (in particular, why moving the `guess_lb >=
current_best` check ahead of the perfect-split shortcut can't wrongly
skip a genuine perfect split) and the measured result: wall-clock neutral
on the tracked salet/full-word-list benchmark once run-to-run noise is
accounted for (see Phase 2b in PROGRESS.md) — kept anyway since it's
provably correct and strictly work-reducing in theory, just not enough to
clear this machine's noise floor on this specific benchmark.

### 2. Phase 3 — disjoint-union bound propagation — DONE

Implemented per the original plan (see PROGRESS.md for the final code
shape and the design-question resolutions below). Verified correct
(full `test_solver.py` including TSan, fixture cross-checks). Measured
impact: near-zero on the tracked benchmark — the union-bound reuse fired
on only 49 of 2,251,108 nodes for salet on the full word list, i.e.
exact-subset collisions across *different* guesses' partial partitions
are rare for this word list's diversity. Kept anyway (real, cheap when it
doesn't fire, may matter more for `--top`/`--all` or smaller word lists
where overlap is more likely) but should not be counted on to close any
of the remaining gap to `wordle.cpp`.

Design questions from the original plan, resolved:
- Hook point: right after a bucket loop's `pruned = true` break (both
  prune sites — the pre-solve admissible-bound check and the post-solve
  `bucket_cost >= bucket_beta` check), using whichever buckets were
  already solved to exact cost earlier in that same iteration.
- Writes go directly via `tt_find_or_claim`, gated so they only ever
  raise `proven_lower_bound` and never touch `exact_cost`, and only when
  `exact_cost` is still unknown for that key.
- TT capacity: `tt_find_or_claim` already degrades gracefully to "stop
  caching" if the table is full (verified this is the existing, already
  load-bearing behavior, not something Phase 3 needed to add) — no
  observed capacity pressure from the extra derived entries at the
  measured 49-hit rate.

### 3. Phase 4 — endgame-cluster cutoff — DONE (static-only, as scoped)

Implemented exactly the scope decision made here: the *static* coverage
cutoff only (re-derived, not copied, from `wordle.cpp:685-709`'s
greedy coverage-counting bound), without the live-endgame fallback
re-search (`wordle.cpp:715-721`). See PROGRESS.md for the full
derivation, the closed-form lower-bound formula (verified against a
brute-force DP before being trusted, not just asserted) and the
independent Python cross-check (589/53 cluster counts matched the C
implementation exactly).

Correctness notes from `gemini_plan.md`'s sketch, both honored:
- No raw `UINT32_MAX` "infeasible" sentinel anywhere — this
  implementation never returns a special value at all; it only ever
  *raises* `node_lb` via `max()` with the generic bound, so it fails
  soft against the caller's `beta` exactly like every other bound in
  `solve_subset` already does, automatically.
- Did not copy wordle.cpp's `minendgamecount=4` gating or gemini's
  `count >= 4` simplification — re-derived instead: a cluster only ever
  gets stored if `branch_capped_lower_bound(size, max_d)` actually
  exceeds the generic `lower_bound[size]`, which self-selects for
  "genuinely constrained" clusters directly (no arbitrary size
  threshold needed beyond skipping size<3, which can be proven useless
  outright: for k<=2, the bound is 2k-1 regardless of branching factor,
  since max_d>=2 always holds whenever a cluster has >=2 members).

**Result: correct, cheap (adds ~0.1s of precompute to a ~200s run), but
measured impact was zero** — see PROGRESS.md for the full measurement
and the "why this has much less leverage than wordle.cpp's own use of
it" analysis (short version: wordle.cpp gets subtree-elimination power
from the live re-search fallback this phase deliberately excluded; the
static cutoff alone only nudges a numeric floor, which rarely lands in
the narrow gap where it flips an actual prune decision for a total-cost
objective).

### 4. Re-measure after Phase 3/4 — DONE

Done as part of landing each phase rather than as a separate pass (see
PROGRESS.md's Phase 2b and the Phase 4 measurement section) — every item
in this plan is now implemented, verified, and measured. The remaining
~6-10x gap to `wordle.cpp`'s 34.25s reference has not been meaningfully
closed by any of items 6/Phase 3/Phase 4; the only unexplored idea left
from wordle.cpp's own playbook is the live endgame re-search fallback,
which was excluded by deliberate scope decision (see item 3 above and
PROGRESS.md) rather than left for later — closing more of the gap would
mean revisiting that decision, not finishing more items from this list.

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
