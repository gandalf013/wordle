# Optimization Progress & Benchmark Log

This document tracks sequential optimization steps, empirical benchmarks, and performance scaling comparisons between `gemini_solver` and Alex Selby's reference solver (`wordle.cpp`).

---

## 1. Head-to-Head Comparison: Alex Selby vs. Gemini Solver

Both solvers run on the same hardware (Apple Silicon 10-core, macOS, clang -O3) against `./words.txt` (**3,209 targets, 14,855 total allowed words**).

### A. Full Corpus Opener Search (3,209 Targets / 14,855 Guesses)

| Benchmark Opener | Exact Total Cost | Alex Selby (`wordle.cpp`) [1 Thread] | Gemini Baseline [1 Thread] | **Gemini Step 1 [1 Thread]** | Gemini Baseline [10 Threads] | **Gemini Step 1 [10 Threads]** | **Step 1 Speedup** | **Current Gap vs. Alex (1-Thread)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`taler`** (Optimal) | **11,483** | **14.84 s** | 66.63 s | **40.67 s** | 35.75 s | **20.17 s** | **1.64× / 1.77×** | Alex is 2.74× faster (was 4.49×) |
| **`salet`** | **11,433** | **34.65 s** | 186.26 s | **107.35 s** | 98.40 s | **76.93 s** | **1.74× / 1.28×** | Alex is 3.10× faster (was 5.37×) |
| **`roate`** | **11,543** | **19.87 s** | 67.08 s | **72.74 s** | 36.12 s | **41.59 s** | ~flat | Alex is 3.66× faster (was 3.38×) |

---

### B. Subset Scaling Comparison (Opener: `aback`)

| Target Count ($N$) | Total Guesses Allowed | Exact Cost | Alex Selby Wall Time | Alex Nodes Used | Gemini Step 1 Wall Time (1 Thread) | Gemini Step 1 Tree Nodes Visited | Relative Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12 (tiny)** | 12 | **34** | 0.3794 s | 5 | **0.0065 s** | **2** | **Gemini is 58× faster** |
| **40 (small)** | 160 | **119** | 0.0061 s | 20 | **0.0041 s** | **7** | **Gemini is 1.49× faster** |
| **120 (medium)** | 480 | **391** | 0.0066 s | 4,773 | **0.0058 s** | **37** | **Gemini is 1.14× faster** |
| **300** | 1,200 | **1,019** | 0.0144 s | 29,577 | **0.0189 s** | **141** | Alex is 1.31× faster |
| **600** | 2,400 | **2,096** | 0.1617 s | 870,443 | **0.2724 s** | **1,605** | Alex is 1.68× faster |
| **1,000** | 4,000 | **3,464** | 0.1105 s | 547,589 | **0.1774 s** | **682** | Alex is 1.60× faster |
| **1,500** | 6,000 | **5,365** | 0.6082 s | 2,768,482 | **1.6883 s** | **3,651** | Alex is 2.78× faster |

---

## 2. Step-by-Step Optimization Log

### Step 1: Multi-Tier Partition Bounding Sieve & Sparse Active Tracking

#### Summary of Architectural Changes:
1. **Tier-1 / Tier-2 Partition Sieve**:
   * Evaluates exact costs for child buckets with $sz \le 2$ analytically ($1$ for $sz=1$, $3$ for $sz=2$).
   * Probes the Transposition Table (`tt_find`) for all buckets with $sz \ge 3$ before performing any recursive calls.
   * If $\text{Tier2\_LB} \ge \text{current\_best}$, immediately discards the candidate guess without recursing into child subtrees.
   * Resolves exact costs immediately without recursion when all child buckets hit exact TT entries or have $sz \le 2$.
2. **Move Ordering by $S_2$ Variance**:
   * Prioritized candidate ranking by $S_2 = \sum \text{size}^2$ ascending, evaluating highest-entropy splits first to establish tight initial $\beta$ bounds.
3. **Sparse Active Score Tracking**:
   * Replaced 243-element `memset` and full array sweeps in candidate loops with active score tracking (`active_scores[]`), resetting only touched entries ($O(|H|)$ vs $O(243)$ writes).

#### Results Achieved:
* **Node Reduction**: Reduced tree nodes visited on opener `taler` from **166,857** to **33,689** (**79.8% node reduction**).
* **Speedup**: 1-thread execution improved from **66.63s $\to$ 40.67s** (**1.64× speedup**); 10-thread execution improved from **35.75s $\to$ 20.17s** (**1.77× speedup**).
* **Sanitizer Verification**: 100% pass rate across plain, ASan, and TSan builds.
