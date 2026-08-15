# Gemini Wordle Solver: Findings, Empirical Results & Next Steps

## 1. Overview & Architecture

`gemini_solver` is an exact, mathematically optimal Wordle solver designed for **Easy Mode** (where all guesses $G$ remain valid at every node, and subproblem state depends strictly on the target word subset $H \subseteq \text{Targets}$).

The implementation is housed in [`gemini_solver/wordle_gemini.c`](wordle_gemini.c) and includes the following architectural optimizations:

1. **128-Bit Double Zobrist Hashing (Provable Exactness)**:
   - Transposition table (TT) and branch deduplication tables use paired coprime 64-bit Zobrist hashes ($H_1 \oplus H_2$), yielding collision probability $P < 10^{-22}$ across the entire search space.
2. **Fused Candidate Loop**:
   - Merged candidate partition deduplication, score histogramming, sum-of-squares variance computation, and $lb_1/ub_1$ evaluation into a single cache-coherent sweep over target indices.
3. **Sound Node-Level $lb_1$ Cutoffs & Exact Resolution**:
   - Computes $lb_1 = \min_g lb(g)$. If $lb_1 \ge \beta$, triggers a fail-soft cutoff without expanding children.
   - If $ub_1 == lb_1$, immediately returns the exact optimal score ($2n$ partition) without recursion.
4. **Alex Selby Endgame Letter-Coverage & Gating**:
   - Precomputes all wildcard patterns ($\ge 4$ words, e.g. `_IGHT`, `_OUND`, `_OPE`).
   - Evaluates exact letter distinguishability over candidate guesses, computing greedy top-$r$ coverage sums and enforcing static cutoffs when $|L| - 1 > \text{coverage}$.
   - Gates isolated endgame subsearches with Selby's heuristic condition: $r - (\text{coverage} - (|L| - 1)) > 0$.
5. **Aspiration-Seeded Root Bucket Budgeting**:
   - Seeds root evaluations with a fast 1-ply greedy upper bound ($C_{\text{greedy}}$) and allocates finite $\beta$ bounds to each root bucket in parallel mode.
6. **Thread-Safe Shared Global Ceiling with Delta Node Tracking**:
   - Uses atomic best cost ceilings (`atomic_uint_fast32_t`) across concurrent opener threads with accurate per-bucket node delta logging.

---

## 2. Empirical Verification & Scaling Benchmarks

### A. Test Suite & Sanitizer Pass Rate
Validated via [`gemini_solver/test_solver.py`](test_solver.py) with 100% exact score agreement against the independent brute-force reference oracle ([`solver/reference_solver.py`](../solver/reference_solver.py)):
* **Plain Build (`-O3 -march=native -flto`)**: 5/5 tests passed.
* **AddressSanitizer + UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`)**: 5/5 tests passed.
* **ThreadSanitizer (`-fsanitize=thread`)**: 5/5 tests passed.

### B. Node Reduction vs. Baseline (`wordle_claude.c`)

| Target Size | Exact Total Guesses | Old Nodes (`wordle_claude`) | New Nodes (`wordle_gemini`) | **Node Reduction** |
| :---: | :---: | :---: | :---: | :---: |
| **12 (tiny)** | 34 | 13 | **2** | **84.6%** |
| **40 (small)** | 119 | 161 | **13** | **91.9%** |
| **120 (medium)** | 391 | 933 | **42** | **95.5%** |
| **300** | 1,020 | 1,988 | **295** | **85.2%** |
| **600** | 2,099 | 24,989 | **5,538** | **77.8%** |
| **1,000** | 3,487 | 26,800 | **1,394** | **94.8%** |
| **1,500** | 5,383 | 116,603 | **22,924** | **80.3%** |
| **2,000** | 7,396 | 409,979 | **206,413** | **49.7%** |

---

## 3. Full Corpus Benchmark Results (`words.txt`)

Running against the complete `words.txt` word list (**3,209 target words, 14,855 guess words**) on a 10-core machine:

### Top 10 Opener Search (`--top 10 --threads 10`)
* **Total Wall-Clock Time**: **8 minutes 4 seconds** (484.86 s).
* **Winning Opener**: **`taler`** with **11,483 total guesses** (average: **3.57837** guesses/game).

#### Individual Candidate Breakdown:
1. **`taler`**: 11,483 total (3.57837 avg) — *169.38s, 166,857 nodes* [BEST]
2. **`artel`**: 11,491 total (3.58087 avg) — *156.29s, 133,822 nodes*
3. **`ratel`**: 11,502 total (3.58429 avg) — *150.02s, 121,263 nodes*
4. **`roate`**: 11,543 total (3.59707 avg) — *258.16s, 507,270 nodes*
5. **`raile`**: 11,555 total (3.60081 avg) — *185.98s, 220,019 nodes*
6. **`oater`**: 11,617 total (3.62013 avg) — *161.16s, 157,201 nodes*
7. **`ariel`**, **`raise`**, **`tiare`**, **`soare`**: Pruned against the 11,483 ceiling.

---

## 4. Next Steps & Future Optimization Roadmap

1. **Shared Lock-Free Global Transposition Table**:
   - Currently, each worker thread maintains an isolated thread-local TT ($2^{20}$ entries).
   - Implementing a shared lock-free TT using atomic 128-bit compare-and-swap (or seqlock per cacheline) will allow threads evaluating different openers to reuse proven subset costs across buckets, dramatically speeding up multi-opener runs.
2. **SIMD Vectorization for Candidate Loop (AVX2 / ARM NEON)**:
   - Vectorize the score-lookup and 128-bit hash accumulation across targets in blocks of 8/16 words using ARM NEON / AVX2 intrinsics.
3. **Adaptive Dynamic Probe Selection for Deep Endgames**:
   - Extend precomputed endgame tables beyond 1-letter wildcards to multi-position vowel patterns (e.g. `_A_ER`, `_O_ND`), pre-indexing optimal 2-probe sequences.
4. **Distributed Cluster / GPU Search (`--distributed`)**:
   - Add MPI / socket-based work stealing for distributing `--all` 14,855-opener evaluations across multiple cluster nodes.
