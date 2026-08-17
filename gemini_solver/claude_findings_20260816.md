# Code Review: `gemini_solver/wordle_gemini.c`

Scope: full read-through of the file (2560 lines), easy-mode Wordle solver. Hard-mode
support is intentionally out of scope per the task. Findings are grouped into
correctness bugs, code quality, and performance.

---

## 1. Bugs / Correctness Issues

### 1.1 [HIGH] `candidate_keys` depth-layer aliasing corrupts in-flight candidate lists (`solve_subset`, lines 1135, 1309-1520)

```c
candidate_keys = solver->candidate_keys + (size_t)(depth < 7 ? depth : 7) * num_guesses;
...
for (c = 0; c < limit; c++) {
    uint32_t clb = (uint32_t)((candidate_keys[c] >> 16) & 0xFFFF);
    ...
    g = (uint32_t)(candidate_keys[c] & 0xFFFF);
    ...
    bucket_cost = solve_subset(solver, &local_partition[unresolved_buckets[u].offset], sz,
                               unresolved_buckets[u].hash1, unresolved_buckets[u].hash2,
                               bucket_beta, NULL, depth + 1);   // line ~1511-1513
    ...
}
```

`solver->candidate_keys` is allocated as 8 fixed layers (`solver_init`, lines 770-771),
indexed by `min(depth, 7)`. Every depth `>= 7` shares **the same** layer. Since
`solve_subset` recurses into its own unresolved buckets with `depth + 1`, a node at
depth 7 that recurses into a child (necessarily also depth `>= 7`, so also layer 7)
will have its own `candidate_keys` buffer **overwritten by the child** while the
parent's `for (c = 0; c < limit; c++)` loop is still mid-flight. The parent has
already consumed `candidate_keys[c]` for the current `c`, but every subsequent
iteration (`c+1`, `c+2`, ...) reads back guess indices (`g`) and lower bounds (`clb`)
that belong to the *child's* search, not its own.

This is a real, actively-recursing single-threaded aliasing bug (not a threading
race) that will happen routinely: Tier-3 recursion (line ~1511) runs for *every*
candidate guess audited at a node, not just the eventual winner, so nodes that reach
depth 7 will very commonly trigger it, especially during `--exhaustive`/large
`--candidates` runs or with bigger custom word lists.

Impact: the corrupted node may skip a candidate that should have been examined (via
the now-garbage `clb`), or examine a nonsensical `g`, silently reducing the search to
a possibly-suboptimal (but still recorded as fail-soft) result. Because the result
gets written to the TT via `solver_tt_store_exact`/`solver_tt_store_lb` (line 1558 /
1564) and the TT is keyed purely by the 128-bit Zobrist hash + size, a single
corrupted computation can poison every other place in the tree (and across openers,
via the shared TT) that reaches the same target subset. This undermines the tool's
core "exact/optimal" guarantee.

**Suggested fix:** don't cap the layer index at all — allocate/track a buffer per
active stack frame (e.g. grow `candidate_keys` layers to a depth that can't be
exceeded within a single search, or malloc/free a `num_guesses`-sized scratch buffer
per call once `depth >= 7`, accepting the extra allocation cost only in the rare deep
case). At minimum, raise the layer count generously (e.g. 32-64) and add an assertion
that `depth` never exceeds it, so a violation fails loudly instead of silently
corrupting results.

### 1.2 [HIGH] Unbounded VLA stack usage risks stack overflow on worker threads (`solve_subset`, lines 1006, 1016, 1020)

```c
uint16_t t_active[count + 1];
...
uint32_t local_partition[count];
...
uint16_t active_scores[count + 1];
```

`count` is the size of the current target subset and is **not bounded** — at the
top of the tree (the first bucket handed to a worker after the opener) it can be in
the hundreds or low thousands, depending on word-list size and opener quality.
Combined with the fixed-size locals in the same frame (`BucketInfo buckets[243]` ≈
7.8 KB, plus several `uint32_t[243]`/`uint8_t[243]` arrays), each `solve_subset`
stack frame can easily be tens of KB when `count` is large.

All solving happens on worker threads created via plain `pthread_create(&t, NULL,
...)` (e.g. lines 334, 1751, 2044, 2503) — i.e. **default thread attributes**. On
macOS, the default stack size for a non-main pthread is only 512 KB (vs. 8 MB for
the main thread on Linux/glibc, which is also not guaranteed). Given `solve_subset`
recurses into itself for every unresolved bucket (Tier 3, line ~1511), and — per
1.1 — that recursion routinely reaches depth 7+ during normal search, it's entirely
plausible to blow the stack on a large first-level bucket combined with several
levels of recursion, producing an unrecoverable crash (SIGSEGV) rather than a
graceful error, especially on macOS or with larger-than-standard word lists.

**Suggested fix:** replace the VLAs with a per-solver scratch buffer sized once to
`game->num_targets` (similar to how `candidate_keys` is pre-allocated in
`solver_init`), or explicitly set a generous stack size via
`pthread_attr_setstacksize` before creating worker threads.

### 1.3 [MEDIUM] `make_leaf` can read out-of-bounds if a target word is duplicated in the "extra guesses" section of the word list (lines 1889-1897, used at 1942, 2054)

```c
static TreeNode *
make_leaf(GameData *game, uint32_t guess_idx)
{
    ...
    strcpy(n->guess, game->targets[guess_idx].word);
    return n;
}
```

`make_leaf` is called both with genuine target indices (`targets[0]`, safe) and with
*guess*-array indices taken from `solve_subset`'s `best_g`/`opener_idx`
(`build_subtree_node_with_guess`, line 1942; `build_solution_tree`, line 2054) when a
guess scores `EXACT_MATCH`. This only works because `load_wordlist` (lines 174-216)
always appends each non-extra line to both `targets` and `guesses` at the same index,
so for `i < num_targets`, `guesses[i] == targets[i]`. An `EXACT_MATCH` guess is
therefore assumed to always have `best_g < num_targets`.

That invariant breaks if the word-list file lists a word both above *and* below the
blank-line separator (i.e. a target word is also repeated in the "extra valid
guesses" section) — a plausible situation for hand-merged or generated word lists.
In that case the duplicate entry gets a `guesses` index `>= num_targets` but still
scores `EXACT_MATCH` against its matching target, so `best_g`/`opener_idx` can end up
`>= num_targets`, and `game->targets[guess_idx]` becomes an out-of-bounds read
(`targets` is only sized `num_targets`).

**Suggested fix:** either de-duplicate the guess list against the target list while
loading, or have `make_leaf` take an explicit target index resolved via a
guess-index→target-index map (or just store the word string directly instead of an
index) rather than relying on the positional-prefix invariant.

### 1.4 [LOW] Silent allocation-failure handling is inconsistent

`load_wordlist`'s `realloc` calls (lines 204, 212) and the vast majority of `malloc`
calls throughout the file (e.g. `part = malloc(...)` in `solve_greedy_tree` line 935,
every `local_partition = malloc(...)` in the opener-evaluation/tree-building paths)
are unchecked, while `init_game_data`'s two big matrix allocations (lines 311-315,
342-346) explicitly check and `exit(1)` on failure. This is a minor consistency nit
rather than a functional bug for typical runs, but on a `realloc` failure in
`load_wordlist` the original buffer pointer is overwritten with `NULL`, and the code
would crash on the next `strcpy` into it rather than reporting "out of memory".

---

## 2. Code Quality

- **`sort64_asc` is not actually an introsort** (file header, line 11, claims
  "Inlined 64-bit Introsort"). It's a median-of-3 quicksort with the standard
  tail-recurse-on-larger-partition trick to bound stack depth, but it has no
  fallback to heapsort on adversarial/degenerate pivots, so its worst case is still
  O(n²) time (just not O(n) stack depth). Harmless in practice since the keys being
  sorted are essentially random rank scores, but the naming/comment overstates what
  it guarantees.

- **Parsed but unused per-word weight** (`load_wordlist`, line 177/189): each line
  may specify `word <weight>`, but `weight` is parsed and then completely discarded.
  Either this is dead functionality (word-frequency-weighted solving was planned but
  never wired in), in which case the field/parsing could be removed, or it's a
  silently-missing feature. Worth clarifying intent.

- **`counts[NUM_SCORES]` naming collision risk**: the function-local `counts` array
  (line 1011) is scratch-reused only inside the `count <= 8` fast path, while the
  `count > 8` path uses a differently-named `big_counts` (line 1213) for the same
  purpose. Not a bug, but the two parallel implementations of the same
  histogram/reset logic (mirrored again a third time in the "target-only" pre-check
  at lines 1069-1104, and a fourth time in the main loop's `hist`/`active_scores`
  bookkeeping at lines 1335-1341) is a lot of hand-duplicated, easy-to-desync logic.
  A single small helper (build histogram + delta-track touched buckets + reset)
  would reduce the surface area for exactly the kind of subtle bug described in 1.1.

- **`pthread_mutex_init(&pool.print_mutex, ...)` is never destroyed** (`main`,
  around line 2498) — harmless since the process exits shortly after, but is a
  resource-hygiene nit, and would matter if this code were ever reused as a library
  entry point instead of a one-shot CLI.

- **Magic numbers**: `16` (TT probe depth, `TT_MAX_PROBES` is defined once for the
  local `TT` but the shared TT's `shared_tt_find`/`shared_tt_store` re-hardcode the
  literal `16` at lines 545 and 583 instead of reusing a named constant). If one is
  ever tuned independently of the other, this is an easy spot to introduce a
  mismatch.

- **`evaluate_opener_parallel` computes `greedy_upper_bound` but the result is
  effectively unused for anything except seeding `bucket_betas`**, and the variable
  is explicitly cast to `(void)greedy_upper_bound;` at line 1785 after already being
  used — the cast is dead since the variable *is* used above; it looks like a
  leftover suppression from an earlier version where it wasn't.

---

## 3. Performance

- **O(count²) "target-only instant resolution" pre-check runs unconditionally at
  every node** (lines 1069-1132), regardless of `count`. For small buckets (which is
  the common case deep in the tree) this is a good, cheap optimization — it's
  actually *cheaper* than the O(num_guesses × count) move-ordering pass that follows
  when `count << num_guesses`. But it is **not gated by a size threshold**, so at
  or near the root (`count` in the hundreds/thousands) this becomes a multi-million
  operation pass that essentially always fails to find a perfect split (a
  same-cost-class Sidon-like condition that's very unlikely to hold for large
  `count`), before the "real" move-ordering computation even begins. Consider
  gating this pre-check behind something like `count <= 64` so large nodes skip
  straight to the guess-based move ordering, which already computes an equivalent
  (better, because it considers the full guess corpus, not just targets) bound via
  `global_ub1`/`best_exact_g`.

- **Full-column reset scan on early-aborted candidates** (`count > 8` branch, lines
  1249-1251):

  ```c
  for (r = 0; r < count; r++) {
      big_counts[cols[r][g]] = 0;
  }
  ```

  This always re-touches all `count` column entries to zero `big_counts`, even when
  the increment loop above it aborted early via `can_abort`/`aborted` (line
  1238-1241) after processing only a handful of targets. The file uses a
  delta-tracking reset pattern elsewhere (e.g. `t_active` in the target-only
  pre-check, lines 1082-1090) specifically to avoid this kind of full rescan —
  applying the same technique here (track only the indices actually incremented
  before the abort, reset just those) would make the early-abort optimization
  actually save the reset cost too, not just the increment cost.

- **TT replacement policy is naive first-victim** (`tt_find_or_claim`, lines
  471-506; `shared_tt_store`, lines 571-620): on a full probe chain, the *first*
  non-matching slot encountered is evicted unconditionally, with no preference for
  keeping already-`exact_cost`-resolved entries over lower-bound-only ones. Under
  memory pressure (large word lists, small `--max-memory`) this can evict expensive,
  fully-proven results in favor of cheap-to-recompute bound entries. A simple
  "prefer evicting the entry without an exact_cost" scan (still O(16), just checking
  a field before committing to a victim) would likely improve TT hit quality without
  meaningfully increasing probe cost.

- **`big_counts[NUM_SCORES]` is `uint16_t`** (line 1213) — safe for typical Wordle
  word-list sizes (thousands of targets), but silently overflows if a single guess's
  bucket for one score exceeds 65535, which is only possible with unusually large
  custom dictionaries (`>~65k` targets). Worth a comment or a `count`-based
  compile-time/runtime bound check if the tool is meant to support arbitrary word
  lists rather than just standard Wordle-scale ones.

- **`solve_greedy_tree` recomputes from scratch with no memoization** (lines
  868-949): it's only used to seed aspiration bounds, so this is a one-time cost,
  not a hot path — noted for completeness but not a priority.
