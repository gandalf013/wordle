# Code Review of `wordle_gemini.c` (2026-08-16)

Review target: [`gemini_solver/wordle_gemini.c`](wordle_gemini.c). Scope: easy-mode Wordle
optimal solver. Line numbers refer to the current file.

Wordlist context used below: `words.txt` = 3209 targets + 14,855 total guesses.

---

## 1. Potential bugs

### 1.1 (P1) `make_leaf` uses the wrong array + out-of-bounds read in tree/JSON export

`make_leaf` (`:1890-1897`) copies from `game->targets[guess_idx]`, but it is called with
**guess indices** in the two exact-match branches:

- `:1942` `make_leaf(game, best_g)` — `best_g` is a guess index
- `:2054` `make_leaf(game, opener_idx)` — also a guess index

The guess and target arrays are different lists (`num_targets=3209`, `num_guesses=14855`).
Consequences:

- If `best_g >= num_targets` → out-of-bounds read of `game->targets` (UB; ASAN would trip).
- If `best_g < num_targets` → still the *wrong word* (`targets[best_g]` is unrelated to
  `guesses[best_g]`).

For the exact-match bucket the leaf's `guess` must be `game->guesses[best_g].word` (the guess
that *is* the answer). The `count == 1` branch at `:1970` correctly passes a target index, which
is why the bug only shows for in-set guesses. This corrupts every `--tree` / `--save-tree` export.

### 1.2 (P2) `in_set` is always false in the ≤8 fast path

In the `count <= 8` loop, `counts[...]` is zeroed at `:1170-1176`, then
`in_set = (counts[EXACT_MATCH] > 0)` is read at `:1207`. So the `-2` in-set bonus in `rank_score`
(`:1208`) never fires, unlike the >8 branch which computes `in_set` *before* clearing
(`:1244` vs `:1249`). Impact is limited to move ordering, but it is a real logic slip — the
sibling `hist[EXACT_MATCH]` bug was fixed earlier, but this one was left behind.

### 1.3 (P2) `good_target` "exact" resolution is not provably exact when the single big bucket has size ≥ 3

`:1106-1120`. When `bad == 1`, the code stores/returns `2 * count` as **exact** if `< beta`.
But `good_target`'s own cost is `2*count + m − 2`, where `m` = size of the one bucket with
≥ 2 targets:

- `m == 2` → cost is exactly `2*count`, and `2*count` is a valid lower bound (no `bad == 0`
  target exists at that point), so "exact" is sound.
- `m >= 3` → `good_target` costs `2*count + m − 2 > 2*count`, and nothing proves some *other*
  guess achieves `2*count`. If none does (degenerate / endgame-cluster data — the same class of
  input the known P2 cluster hang is made of), the stored/returned "exact" understates the node's
  true optimum and poisons the shared TT.

The `>= beta` branch (`:1115`, stored as an LB) is sound; only the `< beta` exact branch is
questionable. Real word data essentially never triggers it; synthetic/adversarial lists can.

### 1.4 (P2/P3, latent) `candidate_keys` layer reuse at recursion depth ≥ 8

`:1135`: layer index is `depth < 7 ? depth : 7` over an 8-layer buffer (`:770`). A call at
depth ≥ 8 reuses layer 7 while its depth-7 parent still needs it — the parent reads
`candidate_keys[c]` for later `c` (`:1310`) after the child has overwritten the buffer.
Depth 8+ is reachable in principle on deep/degenerate searches or larger wordlists; on real data
the search stays shallow, so this is latent, but it is a genuine aliasing hazard. Fix: allocate a
per-stack-frame buffer, or raise the layer cap and assert `depth < layers`.

### 1.5 (P3, by design but worth flagging) Default mode is not exhaustive

`max_candidates = 100` (`:2306`), so every "EXACT RESULT" / `is_exact` claim and the TT
`exact_cost` entries are only optimal within the top-100 candidate guesses unless `--exhaustive`
is passed. Additionally, under this mode two threads can resolve the same node to *different*
"exact" values and race-store them into the shared TT, so runs are not reproducible. Under
`--exhaustive` this is fine (all "exact" values agree).

### 1.6 Minor

- `:1785` `(void)greedy_upper_bound;` is stale — it *is* used for `bucket_betas`.
- `--tree` after a pruned `--opener` (`:2432-2437`) writes `exact_total_guesses = UINT32_MAX`;
  the `--top/--all` fallback (`:2522-2526`) reports the greedy seed as "best" and then dumps it
  as exact.
- `compare_opener_results_asc` (`:2164`) does not break ties → nondeterministic ranking order.

---

## 2. Code quality

- **Duplicated histogram + partition + offset logic** in ~5 places (`partition_root`,
  `build_subtree_node_with_guess`, `solve_greedy_tree`, `compute_opener_greedy_upper_bound`,
  the main loop). A shared helper would reduce the risk of the divergence bugs above (e.g. the
  `in_set` / clearing mismatch).
- **Magic numbers everywhere**: 16 probes, `depth < 7 ? depth : 7`, default 100,
  `1u << 25` / `2^18` / `2^21` / `2^16` caps, `99.0` pruned sentinel, `0.001`.
- Comment claims "Inlined 64-bit Introsort" but `sort64_asc` (`:626-687`) is quicksort +
  insertion sort with no depth-limited heapsort fallback — not actually introsort.
- `weight` parsed via `sscanf` but never used.
- `in_extra` blank-line semantics in `load_wordlist` are undocumented and fragile (a leading
  blank line ⇒ zero targets).
- No OOM checks on most mallocs; `pthread_create` returns unchecked.
- Header's "zero collision risk" is overstated: 128-bit Zobrist XOR hashes are collision-free
  only practically, not provably.
- The 243-entry histogram resets are hand-rolled (`counts[...] = 0` lists) — exactly what led to
  the `in_set` bug. A "touched scores" list would be cleaner and is what the >8 path needs anyway.

---

## 3. Performance

1. **Root-level move-ordering sweep is the hot spot**: O(num_guesses × count) per node. At the
   root (count = 3209, guesses = 14855) that is ~48M iterations per opener eval; `--all` does this
   for 14,855 openers. Mitigations: dedupe identical score rows, or exploit that the shared TT
   makes most root-bucket solves cache hits.
2. **>8-branch histogram clearing** (`:1249-1251`) re-zeroes `big_counts` by scanning *all*
   `count` targets per guess → O(count × G) per node. Use a touched-scores list like the ≤8 path.
3. **`sort64_asc_top`** over 14,855 keys per node is fine with top-k, but `sort64_asc`'s missing
   introsort fallback gives a potential O(n²) worst case on adversarial keys (keys are structured,
   so likely OK, but a heapsort fallback would harden it).
4. **O(count²) `good_target` pre-check** (`:1069-1104`) runs even for count > 243 where `bad == 0`
   is impossible (only 243 scores). Early-exit when `count > 243`.
5. **Tier-2 bucket hash recomputation** (`:1419-1429`) rescans all `count` targets per bucket per
   surviving candidate — O(count × buckets). Usually cheap after tier-1 pruning, but matters for
   count > 243 nodes.
6. **Bucket-hash recomputation in `partition_root` / `build_subtree_node_with_guess`** rescans
   each bucket; caching per-target hashes in a per-node array removes the O(count) rescan.
7. **`--all` seeding**: `compute_opener_greedy_upper_bound` is a full O(G·T·depth) greedy solve per
   opener. Consider computing it lazily or only for the top heuristic openers.
8. **Transposed matrix build** (`:347-351`) is single-threaded over 47.7M cells — trivial to
   parallelize like the forward matrix.

---

## 4. Verified sound (for the record)

- `compute_score` (double-letter / yellow handling).
- `guess_lb` lower-bound math — both branches reduce to `count + Σ(2m−1)` and match
  `lower_bound[]`.
- `s2 == count` perfect-split resolution.
- `ub1 == lb1` exact resolution (only fires for max-bucket ≤ 2, so actual cost equals the LB).
- Fail-soft alpha-beta discipline; tier-2/3 bucket betas; disjoint-union LB merge.
- Lock-free shared TT ordering (`size` published last, readers require `size > 0`).

---

## 5. Recommended order of fixes

1. **#1.1** wrong-word / OOB in tree export — breaks every `--tree` / `--save-tree` output and is
   trivially triggerable with the real wordlist.
2. **#1.2** `in_set` fast-path (one-line fix, restores intended move ordering).
3. **#1.3** `good_target` exact claim — gate the exact store on the big bucket being size 2.
4. **#1.4** candidate-key layer aliasing — per-frame buffer or larger cap + assert.
