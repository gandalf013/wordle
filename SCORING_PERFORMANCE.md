# Wordle Scoring Performance & Architecture

This document details the design, algorithms, hardware-level considerations, and benchmarks for Wordle score computations in this repository.

---

## 1. High-Level Summary

Wordle scoring (evaluating a `guess` word against a `target` word to yield a 5-trit base-3 score $0..242$) is the core computational primitive during candidate pool narrowing and strategy evaluation (e.g. `EntropyStrategy` or `TwoPlyExpectimaxStrategy`).

Round 1 against the full word list requires scoring $14,855 \text{ guesses} \times 3,209 \text{ targets} = 47,669,695$ pairs. Computing the full guess-by-guess score matrix requires $14,855 \times 14,855 = 220,671,025$ pairs.

### Verified raw scoring benchmarks (Apple M5, 4P+6E cores, `uv run python src/benchmark_scoring.py`)

| Scoring Mode | 14.8k x 3.2k Matrix (Round 1 Pool) | 14.8k x 14.8k Matrix (Full Pool) | Throughput | Speedup vs Original |
| :--- | :--- | :--- | :--- | :--- |
| **Scalar Python** (`scoring.get_score`) | ~62 s (extrapolated) | ~287 s (extrapolated) | 0.77 M pairs/sec | 1x |
| **NumPy Vectorized** | 2.78 s | 12.9 s | ~17 M pairs/sec | ~22x |
| **C Accelerated** (Multi-threaded) | **0.057 s** | **0.25 s** | **~830-870 M pairs/sec** | **~50x (vs NumPy) / ~1,100x (vs scalar)** |
| **Fused C Score + Bincount** | **0.052 s** | -- | **~915 M pairs/sec** | Fused 1-Pass |

These reproduce the previous commit's headline claims (the numbers above were independently re-measured, not copied) -- **the raw C scoring kernel is correctly implemented and genuinely ~50x faster than the NumPy fallback.** Run-to-run variance on this machine is roughly ±10%; treat single-decimal precision in either direction as noise, not regression.

### The catch: raw scoring speed is not what limited real gameplay

The benchmarks above measure `score_matrix` over the full range of both axes shrinking together (a synthetic "how fast can we score N x M pairs" test). **That is not the actual shape of the work `SolverEngine` does.** In real play:

- The **guess list `G` never shrinks** during a game -- every round scores against the same ~14,855 words (or ~3,209 for the weighted list).
- Only the **candidate pool `T` shrinks** round over round, often down to single digits by round 3+.

So the realistic per-round workload is "G fixed at ~15k, T shrinking from ~3,209 down to ~5" -- not "both axes shrinking together." Profiling `analysis.analyze_all` (what `SolverEngine.get_analyses` actually calls every round) against that realistic shape showed the C scoring kernel accounted for **under 10% of wall-clock time** in the common case. The other 90%+ was:

1. **A NumPy aggregation pass over a `(G, 243)` array** (`entropy`/`worst_case_size`/`expected_size`/weighted variants), done as ~10 separate single-threaded `np.where`/`log2`/`sum` calls. That array is 6+ MB at `G=3,209` and ~29 MB at `G=14,855` -- several full passes over an array that size, single-threaded, is memory-bandwidth-bound and **did not depend on `T` at all**, so it cost the same ~5-25ms whether the candidate pool was 5 words or 3,209.
2. **Constructing one `GuessAnalysis` Python object per guess, every round** (~15k or ~3.2k dataclass instances), regardless of how small the candidate pool had shrunk to.

Neither of these is "scoring" in any sense the original benchmark measured -- they're downstream aggregation/object-construction cost that scales with the guess list size, not the target pool size.

---

## 2. What changed in this pass

### Fused stats into the C kernel (`fast_scoring.score_and_analyze` / `ScoringStats`)

The multithreaded C kernel already built each guess's 243-bucket count/mass row while it was hot in that worker thread's L1 cache. `score_bincount_stats_parallel` (in `fast_scoring.py`) now reduces that row to `entropy`/`worst_case_size`/`expected_size` (and the weighted variants) **in the same pass**, instead of handing a `(G, 243)` array back to NumPy for a second, single-threaded, multi-pass reduction. `analysis.analyze_all` was rewritten to consume these pre-reduced arrays directly.

This eliminates essentially all of finding #1 above. A pure-NumPy fallback (`_stats_from_counts_masses`) implements the identical math for the `HAS_C_LIB=False` case and for `use_cache=True` reuse of a disk-cached matrix (recomputing the reduction from a cached matrix isn't worth a second C entry point, since `use_cache` only fires once per process, for round 1).

### Adaptive thread count for small jobs (`fast_scoring._thread_count_for`)

`pthread_create`/`join` has fixed overhead (tens of microseconds each on this machine's heterogeneous P+E cores) that dominates wall-clock time for small jobs. Measured directly: scoring a 10-word candidate pool against the full guess list took *longer* with 10 threads (0.39ms) than with 4 (0.38ms) -- thread setup cost more than the ~150k pairs took to score. Thread count now scales with total pair count (`~20k pairs/thread`, capped at `NUM_THREADS`), so small rounds don't pay full thread-pool setup cost, and large rounds are unaffected (already thread-count-optimal at the max).

### Verified NOT worth doing (ruled out empirically, not left untested)

- **`-mcpu=native` / `-mcpu=apple-m1` compiler flags**: no measurable difference vs. plain `-O3` (within noise, ±5%). Apple clang already targets a NEON-capable baseline for `arm64-apple-darwin`, unlike x86 where `-march` gates SSE/AVX; there's no gate here to unlock.
- **Skipping the `uint8` matrix write when only bucket counts are needed**: measured <1% difference (196.2ms vs 194.9ms on the full 220M-pair matrix). The store is cheap relative to the score computation; not worth the code path split it would require in the hot loop.
- **More threads than physical cores**: scaling plateaus at 10 threads (this machine's physical core count) and going to 12/16 makes no difference either way -- the OS scheduler handles oversubscription fine here, so no need to special-case it down.

### Net effect (measured, `src/benchmark_strategies.py --sample-size 100`)

| Strategy | Before | After | Speedup |
| :--- | :--- | :--- | :--- |
| 1-Ply EntropyStrategy | 1.47s (0.015s/game) | 0.73s (0.007s/game) | ~2.0x |
| 1-Ply ExpectedPoolSizeStrategy | 1.46s (0.015s/game) | 0.71s (0.007s/game) | ~2.1x |
| 2-Ply TwoPlyExpectimaxStrategy | 2.76s (0.028s/game) | 1.97s (0.020s/game) | ~1.4x |

Identical win-rate statistics before/after (simple avg, weighted avg, worst case, games-over-6 all unchanged) -- confirms the change is a pure performance fix, not a behavior change. See `tests/test_fast_scoring.py::TestScoreAndAnalyze` for correctness pinning against an independent NumPy reimplementation of the reduction math, for both the C path and the fallback.

*(The absolute times in this table were measured on the word list as it existed when this pass landed, and have since drifted as the list grew to 3,209 targets; the speedup ratios are the meaningful result. See §1 for current raw-scoring figures.)*

---

## 3. What's still the bottleneck, and what to do about it

After the fix above, profiling `analyze_all` at a realistic size (`G=14,855`, `T=5`) shows the remaining time split roughly:

- ~9% in `score_and_analyze` (scoring + fused stats -- the part this pass optimized)
- ~50% in `GuessAnalysis.__init__` (building one dataclass instance per guess)
- ~37% in `analyze_all`'s own per-guess loop body (dict/set lookups, list appends, kwarg marshaling)

**This is now the dominant cost, and it's architectural, not a kernel problem.** `analyze_all` builds a `GuessAnalysis` object for *every* guess in the list (~15k), every round, even though `SolverEngine.suggest()` only ever reads the *first* one after ranking, and the REPL's `top N` command only reads the first `N` (typically ≤10). The scalar fields it needs to rank by (`entropy`, `expected_size`, etc.) are already sitting in flat NumPy arrays inside `ScoringStats` before the Python loop ever runs -- ranking could be done with `np.argsort`/`np.argpartition` directly on those arrays, and `GuessAnalysis` objects constructed lazily only for the handful actually consumed.

This was **not implemented in this pass** because it changes the `Strategy` protocol's contract (`rank(analyses: list[GuessAnalysis]) -> list[GuessAnalysis]`), which `cli.py`, `display.py`, and every strategy in `strategies.py` depend on -- including `EntropyStrategy`'s documented tie-break semantics that deliberately preserve an "accidental artifact" of `np.argsort` reversal from the original `Game.find_best_guess`. That's a real design decision (struct-of-arrays ranking + lazy object construction vs. today's list-of-objects), not a drop-in optimization, so it's flagged here as the clear next step rather than pushed through unreviewed. Rough expected payoff: eliminating ~85-90% of `analyze_all`'s remaining cost for the common case (small `T`, which is most rounds after round 1) -- i.e. another ~5-8x on top of this pass's ~2x, for the rounds that dominate total game-playing time.

### Other directions considered and deprioritized

- **Explicit SIMD (NEON intrinsics) rewrite of `score_pair`**: the current C loop is already saturating ALU throughput at ~830-870M pairs/sec across 10 threads on this hardware; a hand-vectorized rewrite (batching multiple targets per guess into NEON registers) could plausibly find another 2-4x on the *scoring* step specifically, but scoring is no longer the bottleneck in real gameplay (see above) -- it only matters for the once-per-process round-1 computation and full G×G matrix generation, both already sub-250ms. Worth revisiting only if a future workload (e.g. deep multi-ply search materializing many full-pool matrices) makes raw scoring the bottleneck again.
- **Persistent thread pool** (vs. `pthread_create`/`join` per call): would shave the remaining ~0.2-0.3ms fixed overhead per call at small `T`, but the adaptive thread-count fix already captured most of that win at a fraction of the complexity/risk (no lifecycle management, no thread-safety concerns from a long-lived pool shared across calls).
- **`float32` instead of `float64` for counts/masses**: would halve the memory-bandwidth cost of the (now largely eliminated) NumPy aggregation pass; moot after fusing that pass into the C kernel, where the row lives in registers/L1 regardless of width.

---

## 3a. Portability (Linux / Windows)

The scoring algorithm itself (`score_pair`) is fully architecture-independent: no intrinsics, no `__builtin_*` calls, no endianness/word-size assumptions. It compiles identically on x86_64, arm64, or anything else gcc/clang targets.

**Threading is the real cross-platform constraint.** The kernel uses `pthread.h`, which is standard on Linux/macOS but not available under MSVC (Windows's default toolchain) -- it works on Windows only via MinGW-w64, WSL, or Cygwin. `_load_c_lib()`'s compiler search (`shutil.which("clang") or shutil.which("gcc")`) typically finds nothing on a stock Windows machine, and every failure path (missing compiler, compile error) is caught and falls back to `HAS_C_LIB = False` -- the pure-NumPy path, still correct, ~20x slower than the C kernel. **Nothing crashes; Windows without a POSIX-capable toolchain just doesn't get the C acceleration.** Closing that gap for real (without requiring the user to install MinGW) would mean dropping the pthread dependency from the C side entirely and doing the parallel fan-out in Python via `concurrent.futures.ThreadPoolExecutor` instead (ctypes calls release the GIL for their duration, so this preserves true multi-core speed) -- not done in this pass; flagged here as the next step if native-Windows acceleration becomes a priority.

**`-march`/`-mcpu` tuning is now applied, machine-appropriate, and cache-safe.** `_arch_tuning_flag()` picks `-march=native` on x86_64 (unlocks AVX2 -- the ARM64 finding above of "no measurable win" doesn't generalize to x86, whose baseline ISA is SSE2-only and needs an explicit opt-in for wider SIMD) or `-mcpu=native` on arm64/aarch64 (matches this doc's measured ARM64 result: harmless, no expected win, kept for build consistency). This is only safe because the library compiles at runtime *on the machine that will run it*, not as a distributed binary -- so the compiled cache is now keyed on `hostname + arch + source`, not just source, meaning a `src/.wordle_cache/` directory copied to a different machine (e.g. via `rsync` instead of a fresh clone) always triggers a fresh, correctly-tuned recompile rather than risking an illegal-instruction crash from reusing a stale native-tuned binary. If the compiler rejects the tuning flag for any reason, compilation retries once with baseline flags before giving up. Also added: an explicit `-pthread` compile flag (macOS links pthread symbols into `libSystem` unconditionally so this was a silent no-op there, but musl/Alpine and some glibc configurations need it to link correctly).

The x86_64 `-march=native` win is expected but **unverified in this pass** -- no x86 hardware was available to measure it directly; the ARM64 numbers throughout this document were re-measured with the new flags and are unchanged (as expected, since ARM64 had no gap for `-mcpu` to close).

---

## 4. Architectural Design

```
                     +---------------------------------------+
                     | analysis.analyze_all() / SolverEngine |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |     fast_scoring.score_and_analyze     |
                     +---------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
              (HAS_C_LIB = True)                  (HAS_C_LIB = False)
                       v                                   v
        +----------------------------+      +----------------------------+
        | score_bincount_stats_      |      |  score_matrix_and_bincounts |
        | parallel: score + bincount |      |  + _stats_from_counts_masses|
        | + entropy/expected/worst-  |      |  (NumPy multi-pass          |
        | case fused in ONE          |      |   reduction, same math)     |
        | multithreaded C pass       |      |                              |
        +----------------------------+      +----------------------------+
```

### Auto-Compilation & Runtime Loading
- `fast_scoring.py` automatically compiles an inline POSIX-multithreaded C routine upon first load using system `clang` or `gcc`.
- The compiled shared object (`libscore_<hash>.dylib/.so/.dll`) is cached in `src/.wordle_cache/`.
- If no system C compiler is installed or compilation fails, `fast_scoring.py` sets `HAS_C_LIB = False` and falls back transparently to an optimized NumPy path without raising errors.

---

## 5. Algorithm & Bitwise Mechanics

Wordle scoring uses a 2-pass algorithm for length-5 words:

1. **Green Pass**:
   Position $i \in \{0..4\}$ is GREEN (2) if $g[i] == t[i]$.
2. **Target Counts & Non-Green Remaining**:
   Count remaining unmatched target occurrences for letter $L = g[i]$:
   $$\text{rem}[i] = \sum_{j=0}^4 \mathbb{I}(g[i] == t[j] \land \text{green}[j] == 0)$$
3. **Yellow Pass**:
   Iterate $i \in 0..4$ left-to-right:
   If $\text{green}[i] == 0$ and $\text{rem}[i] > 0$, position $i$ is YELLOW (1). Decrement $\text{rem}$ for all subsequent positions in $g$ that share letter $g[i]$.
4. **Base-3 Encoding**:
   $$\text{score} = s_0 \cdot 81 + s_1 \cdot 27 + s_2 \cdot 9 + s_3 \cdot 3 + s_4$$

---

## 6. Usage & Benchmarks

### Running the Scoring Benchmark
```bash
uv run python src/benchmark_scoring.py
```
Section 5 of its output (`analyze_all at realistic in-game pool sizes`) is the one that reflects actual `SolverEngine` cost -- sections 1-4 measure raw `score_matrix` throughput only, which is informative for round-1/full-matrix cases but not representative of a typical mid-game round.

### Running Strategy Benchmarks
```bash
uv run python src/benchmark_strategies.py --sample-size 100
```

### Running Tests
```bash
uv run pytest
```
