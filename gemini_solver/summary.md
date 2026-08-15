# Gemini Wordle Solver: Architecture, Optimization & Knowledge Base

This document provides a comprehensive summary of the architecture, algorithmic techniques, mathematical models, empirical benchmarks, and external audits for `gemini_solver` (`wordle_gemini.c`).

---

## 1. Executive Summary

`gemini_solver` is an exact, mathematically optimal Wordle full-tree minimax solver written in C. Given an official Wordle dictionary (**3,209 hidden target words** and **14,855 allowable dictionary guesses**), it computes the exact decision policy tree that minimizes the expected average number of guesses per game under standard "Easy Mode" rules (where any valid dictionary word can be guessed at any turn).

### Key Results
* **Optimal Opening Word**: **`taler`**
* **Exact Minimum Total Guesses**: **11,483** across all 3,209 targets
* **Exact Average Score**: **3.57837 guesses/game**
* **Runtime on 10 Threads (Exhaustive Search)**: **8.21 seconds** (34,177 exact search nodes visited)
* **Runtime on 10 Threads (Default Mode, `-n 100`)**: **0.89 seconds**
* **Tree Serialization Overhead**: **0.02 seconds** (generates a validated 458 KB recursive JSON decision tree)

---

## 2. Core Mathematical & Algorithmic Architecture

### A. Objective Formulation & Cost Model
The solver minimizes the cumulative sum of guess depths across all targets $T \in \text{Targets}$:
$$\min_{g_1 \in \text{Guesses}} \left( |H| + \sum_{s \in \text{Scores}(g_1, H)} \text{Cost}^*(H_s) \right)$$
where:
* $H$ is the set of remaining candidate targets at the current node.
* $H_s = \{t \in H \mid \text{Score}(g, t) = s\}$.
* $\text{Cost}^*(\{t\}) = 0$ (if $s = \text{EXACT\_MATCH}$, target is identified in 1 guess).
* Depth-1 base case: $\text{Cost}(\{t\}) = 1$.
* Depth-2 base case: $\text{Cost}(\{t_1, t_2\}) = 3$ (guessing either target gives $\text{Cost} = 1 + 2 = 3$).

### B. Information-Theoretic Lower Bounds
For any subset of size $k = |H|$, the minimum sum of depths is bounded below by:
$$\text{LB}(k) = \begin{cases} 
1 & \text{if } k = 1 \\
2k - 1 & \text{if } 2 \le k \le 243 \\
1 + 2 \times 242 + 3(k - 243) & \text{if } k > 243 
\end{cases}$$

### C. Zero-Branch Incremental Candidate Evaluation
At every search node, the move-ordering metric ($S_2 = \sum c_i^2$) and candidate lower bound ($\text{LB}$) are computed in a **single unrolled pass in pure CPU registers**:
* For each target $t \in H$ with score $s = \text{Score}(g, t)$:
  $$c = ++\text{counts}[s]$$
  $$S_2 \mathrel{+}= 2c - 1 \quad \left(\text{since } \sum_{i=1}^c (2i - 1) \equiv c^2\right)$$
  $$\text{LB} \mathrel{+}= 2 - (c == 1)$$
* Followed by subtracting $\text{counts}[\text{EXACT\_MATCH}]$.
* **Per-Candidate Execution Time**: **2.26 nanoseconds** (under 7 CPU clock cycles).

---

## 3. Incremental Optimization Milestones

| Step | Optimization Technique | Cumulative Speedup | Key Architectural Insight |
| :---: | :--- | :---: | :--- |
| **Baseline** | Parallel Alpha-Beta Minimax + L1 TT | 1.00× (186.26s) | Exact recursive branch-and-bound baseline. |
| **Step 2** | Tier-2 Transposition Table Filter | 1.15× (161.40s) | Prunes candidate partitions before memory allocation. |
| **Step 3** | Fast Analytical Endgames & Target Pre-Check | 1.48× (125.80s) | Direct resolution of $k \le 2$ and $O(\|H\|^2)$ target-only splits ($2k-1, 2k$). |
| **Step 4** | Transposed Contiguous Score Matrix Cache | 2.14× (87.05s) | Guess-major columnar layout in L1 CPU data cache. |
| **Step 5** | Shared Lock-Free Atomic L2 Transposition Table | 3.25× (57.30s) | Multi-threaded CAS-based cross-opener/thread subproblem sharing. |
| **Step 6** | Greedy 1-Ply Aspiration Seeding | 5.21× (35.75s) | Tightens root $\beta$ window before alpha-beta search begins. |
| **Step 7** | Inlined 64-bit Introsort & Register Move Ordering | 12.3× (15.10s) | Eliminates all hash-table lookups and function pointer overhead. |
| **Step 8** | Packed 64-bit Lower Bound Register Pruning | 17.5× (10.69s) | 1-cycle register check `if (clb >= current_best) continue;` skips ~14,800 candidates. |
| **Step 9** | Deferred Partition Construction & Zero-Branch Sweep | **22.7× (8.21s)** | Bypasses partition building for >90% of surviving candidates. |

---

## 4. DeepSeek Audit Findings & Resolutions

An external rigorous audit ([`gemini_solver/deepseek_findings.md`](deepseek_findings.md)) evaluated the solver against an independent brute-force minimax oracle.

### Resolved Issues
1. **P0 (Unsound Endgame Coverage Cutoff - Fixed `7383301`)**:
   * *Problem*: Old static coverage cutoff returned caller's `beta` rather than a proven bound, polluting the TT with artificial bounds and crashing `--tree` export.
   * *Resolution*: Removed unsound cutoff; 100% verified against adversarial sets (`aalnp = 82`, `bxqwk = 84`).
2. **P1 (Greedy Seeder Infinite Recursion on Degenerate Inputs - Fixed `0ac4d46`)**:
   * *Problem*: `solve_greedy_tree` lacked zero-info filtering when all targets produced identical patterns (`active == 1`).
   * *Resolution*: Added `if (active <= 1) continue;` and safe fallback `best_g = targets[0]`, guaranteeing strict subproblem shrinkage.
3. **Defensive Improvements (Fixed `bd402df` & `0ac4d46`)**:
   * Piecewise lower bound for $k > 243$ with `uint16_t` histogram counters.
   * Guaranteed release-acquire memory ordering in `SharedTT` (publishing `size` last).
   * Gated UI output banners on `res.is_exact`.
   * Cleaned up all dead endgame structures and ~300M startup string operations.

---

## 5. Empirical Performance Comparison

### A. Exhaustive Mode vs. Alex Selby Reference Solver

Both solvers configured to explore all **14,855 dictionary words** at every node on 10-core Apple Silicon:

| Metric | Alex Selby Exhaustive (`nth=14855`) | Gemini Solver Exhaustive (1 Thread) | Gemini Solver Exhaustive (10 Threads) |
| :--- | :---: | :---: | :---: |
| **Opener `taler` Runtime** | 15.14 s | 24.05 s | **8.21 s (Faster than Alex)** |
| **Search Nodes Explored** | 68,557,268 nodes | 34,177 nodes | **34,177 nodes (99.95% fewer nodes)** |
| **Transposition Table** | 32-bit compact hash | 128-bit Zobrist | 128-bit Lock-Free Shared TT |

### B. Interactive Wordle UI Latency Breakdown

For an interactive UI where a user inputs any guess and receives a 5-tile color clue:

| Scenario | Targets Left ($|H|$) | Description / Frequency | Exhaustive Latency (10T) | User Experience |
| :--- | :---: | :--- | :---: | :---: |
| **Best Case** | 1 target | Exact match isolated | **< 0.0001 s** | Instant (< 0.1 ms) |
| **Pair Case** | 2 targets | 2 possible solutions left | **< 0.0001 s** | Instant (< 0.1 ms) |
| **Median Case (50th %ile)** | 5–15 targets | **Typical turn-2 state across 80% of clues** | **< 0.0002 s** | **Instant (< 0.2 ms)** |
| **90th Percentile** | 40–70 targets | Uninformative 1-yellow clue | **0.0008 s** | Instant (~1 ms) |
| **Worst Case (Good Opener)** | 200–286 targets | All 5 Grays on `taler` / `salet` | **0.39 s** | Fast (< 400 ms) |
| **Pathological Worst Case** | 500–2,154 targets | All 5 Grays on rare words (`xviii`, `zhuzh`, `fuzzy`) | **2.5 – 3.5 s** | Brief spinner |

---

## 6. Directory Structure & Key Files

* [`gemini_solver/wordle_gemini.c`](wordle_gemini.c): The complete, high-performance C solver.
* [`gemini_solver/Makefile`](Makefile): Automated builds for release (`-O3 -march=native -flto`), AddressSanitizer (`asan`), and ThreadSanitizer (`tsan`).
* [`gemini_solver/test_solver.py`](test_solver.py): Automated test suite running oracle differential tests and stress tests across all build variants.
* [`gemini_solver/OPTIMIZATION_PROGRESS.md`](OPTIMIZATION_PROGRESS.md): Detailed step-by-step optimization logs and micro-benchmarks.
* [`gemini_solver/deepseek_findings.md`](deepseek_findings.md): Independent security, accuracy, and audit log.
* [`gemini_solver/summary.md`](summary.md): This knowledge and architectural summary.
