# DeepSeek Review of `wordle_gemini.c`: Accuracy, Bugs, and Efficiency

Review target: [`gemini_solver/wordle_gemini.c`](wordle_gemini.c). All results were verified
empirically against an independent brute-force minimax oracle and by patching the source and
re-running.

**Status (4th review):** the alpha-beta engine is exact, and every correctness bug found so far
(P0 coverage cutoff, P1 greedy infinite recursion) is **fixed and verified**. All minor items are
resolved. The **only remaining item is the P2 exponential blowup in `solve_subset` on
endgame-cluster-like inputs** (a hang on crafted input; not reachable with real word data).

---

## 1. Accuracy

### What is verified correct

* **Scoring & cost model.** `compute_score` matches the standard Wordle algorithm, and the
  objective (minimize total guesses, i.e. average) is applied consistently throughout.
* **`lower_bound[]` table** (`:194-203`): `lb(1)=1`, `lb(k≤243)=2k−1`, else `1+2·242+3(k−243)` —
  a valid information-theoretic lower bound on the sum-of-depths objective.
* **Target-only pre-check** (`:655-694`): `bad==0 ⇒ 2k−1` and `bad==1 ⇒ 2k` early resolutions are
  both sound (a perfect split requires a target guess that *is* an exact match).
* **Alpha-beta discipline.** Every return path returns either an exact value `< beta` or a
  fail-soft value `≥ beta` (callers check `cost >= bucket_beta` before trusting it). Tier-2 TT
  cutoffs, `global_lb1` cutoffs, `ub1==lb1` resolution, and the disjoint-union bound propagation
  are all sound.
* **128-bit Zobrist TT** is well-ordered and effectively collision-free; the shared lock-free TT
  is correctly synchronized (size published last; readers require `size > 0`).

### FIXED (P0): endgame coverage cutoff was unsound (commit `7383301`)

The old code returned the caller's `beta` (instead of a proven bound) from `solve_subset`,
corrupting `--opener` totals, `--top/--all` results, and crashing `--tree` via an unset
`out_guess`. Reproduced vs brute force and verified fixed:

| Mode | Stock (buggy) | Fixed (`7383301`) | Brute-force truth |
|---|---|---|---|
| `--opener aalnp --exhaustive` (parallel) | **83** | **82** | 82 |
| `--opener bxqwk --exhaustive` | **85** | **84** | 84 |
| `--top 1 --exhaustive` (sequential) | **4294967295** | **82** | 82 |
| `--tree` dump | **SIGBUS / ASan stack-overflow** | **OK** | — |

Same commit also fixed the `hist[EXACT_MATCH]` reset bug in the fast path (the sweep now tests
`scN == EXACT_MATCH` directly instead of reading an already-zeroed histogram) and made `best_g`
assignment unconditional.

### FIXED (P1): `solve_greedy_tree` infinite recursion / stack overflow (commit `0ac4d46`)

`solve_greedy_tree` (the greedy upper-bound seeder) lacked the zero-info guard that `solve_subset`
has. When the greedy's chosen guess (on a sum-of-squares tie it picks the first guess) had already
been removed from the current subset, every remaining target scored the same pattern → a single
bucket of size == `count` → infinite recursion → stack overflow (SIGSEGV on synthetic clusters
with n ≥ 90).

**Fix (verified):** the pick loop now skips zero-info guesses (`if (active <= 1) continue;`) with a
safe fallback (`best_g = targets[0]`, always a member of the current subset → exact match →
progress). The patched build no longer crashes on cluster inputs (they now time out instead — see
P2 below) and still matches the brute-force oracle on all differential + adversarial tests.

### Caveat: default mode is NOT exhaustive (by design)

`max_candidates` defaults to 100; results are only guaranteed optimal with `--exhaustive`. This is
by design and not treated as a defect.

---

## 2. Remaining item (P2): exponential blowup in `solve_subset` on cluster inputs

A set of k targets where every guess separates at most 2 of them (the guessed word + the rest)
makes `solve_subset` explore ~all 2^k subsets. Synthetic clusters of n ≈ 60+ targets (e.g. all
`letter+"zzxy"`, differing at one position) therefore **hang indefinitely** in both `--opener`
and `--top/--all` modes.

* This is a performance/robustness limit, not a correctness bug — given unbounded time the search
  would terminate with the exact optimum.
* It is inherent to exact solving of such degenerate sets, and is exactly what the deleted endgame
  analysis was designed to accelerate. Mitigating it would require re-introducing
  cluster-aware reasoning or a different decomposition.
* **Not reachable with real word data:** real endgame clusters are ≤ ~26 words and are split by
  real guesses, so the search terminates normally (verified: 260-target random real-word set
  completes in well under a second).

---

## 3. Fixed minor items (all resolved)

1. **`is_exact` in `evaluate_opener_parallel`** — detects fail-soft buckets
   (`if (pool.out_costs[b] >= bucket_betas[b]) failed = true;`) and reports `is_exact = !failed`.
   Defensive (cannot fire in practice — a bucket value `≥ beta` would contradict the greedy
   bound's achievability).
2. **Slow-path lb under-estimate for buckets > 243** — now exact: `uint16_t big_counts[]` and the
   piecewise increment `(c == 1) ? 1 : (c <= 243 ? 2 : 3)` reproduce `lower_bound[sz]` for all
   sizes.
3. **Theoretical `shared_tt_store` race** — addressed: `size` is published last and readers
   require `esz == size && size > 0`. Residual window (stale `hash2`/`size` matching a colliding
   `hash1`) is astronomically unlikely.
4. **Dead endgame tables** — removed entirely; one-time startup cost gone.
5. **Stale header comment** — updated to reflect the current feature set.
6. **`main` `--opener` banner** — now gated on `res.is_exact` ("EXACT RESULT" vs
   "PRUNED RESULT"), so it can no longer print `4294967295` under "Exact Total Guesses".

---

## 4. Efficiency

* **Memory:** score matrix + transpose ≈ 220 MB each; per-thread local TT ≈ 64 MB
  (2^21 × 32 B) each; shared TT ≈ 128 MB (2^22 × 32 B). Several GB total at 16 threads.
* **Hot spot:** every node sorts **all** 14,856 candidate keys (`sort64_asc`,
  ~O(G log G) ≈ 200k comparisons) even though only the top candidates are explored — the most
  likely bottleneck.
* **Per-frame stack at large counts:** `solve_subset` places `local_partition[count]`,
  `cols[count]`, `t_active[count]`, `active_scores[count]` on the stack (~16·count bytes/frame,
  up to ~240 KB at count = 14,856). Recursion depth is small for real data.
* **Good:** transposed matrix gives contiguous per-guess columns at small counts; the incremental
  s2/lb sweep is single-pass per candidate; in-register Zobrist bucket hashing (Tier 2) avoids
  materializing partitions until a candidate survives; fail-soft TT with max-merging of lower
  bounds; well-built lock-free shared TT.

---

## 5. Reproduction

### Historical (P0, fixed in `7383301`)

14-target adversarial set (`bxqwk..txqwk + aalnp`; 13 words share pattern `.xqwk`):

```
./wordle_gemini --wordlist adversarial.txt --opener aalnp --exhaustive --threads 8     # was 83 (truth 82)
./wordle_gemini --wordlist adversarial.txt --top 1 --exhaustive --threads 8           # was 4294967295
./wordle_gemini --wordlist adversarial.txt --opener aalnp --exhaustive --tree out.json # was SIGBUS
```

### Historical (P1, fixed in `0ac4d46`)

```
./wordle_gemini --wordlist c90.txt --opener azzxy --exhaustive --threads 1   # was SIGSEGV (stack overflow)
```

### Remaining (P2, open)

Generate an n-word cluster (n ≥ 60) and run:

```python
targets = [(chr(ord('a') + i) + "zzxy") for i in range(60)]
with open("c60.txt", "w") as f:
    f.write("\n".join(targets) + "\n\n" + "\n".join(w + " 1" for w in targets) + "\n")
```

```
./wordle_gemini --wordlist c60.txt --opener azzxy --exhaustive --threads 1   # hangs (P2, exponential)
```
