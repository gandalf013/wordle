# Wordle Scoring Performance & Architecture

This document details the design, algorithms, hardware-level considerations, and benchmarks for Wordle score computations in this repository.

---

## 1. High-Level Summary

Wordle scoring (evaluating a `guess` word against a `target` word to yield a 5-trit base-3 score $0..242$) is the core computational bottleneck during candidate pool narrowing and strategy evaluation (e.g. `EntropyStrategy` or `TwoPlyExpectimaxStrategy`).

Round 1 against the full word list requires scoring $14,855 \text{ guesses} \times 2,309 \text{ targets} = 34,300,195$ pairs. Computing the full guess-by-guess score matrix requires $14,855 \times 14,855 = 220,671,025$ pairs.

### Key Benchmark Results

| Scoring Mode | 14.8k x 2.3k Matrix (Round 1 Pool) | 14.8k x 14.8k Matrix (Full Pool) | Throughput | Speedup vs Original |
| :--- | :--- | :--- | :--- | :--- |
| **Scalar Python** (`scoring.get_score`) | ~44.5 s | ~286 s | 0.77 M pairs/sec | 1x |
| **NumPy Vectorized** (Original) | 3.11 s | 20.32 s | 11.03 M pairs/sec | 14x |
| **NumPy Vectorized** (Optimized Fallback) | 1.98 s | 12.96 s | 17.34 M pairs/sec | 22x |
| **C Accelerated** (Multi-threaded) | **0.038 s** | **0.223 s** | **987.61 M pairs/sec** | **58x (vs NumPy) / 1,280x (vs scalar)** |
| **Fused C Score + Bincount** | **0.045 s** | **0.260 s** | **756.71 M pairs/sec** | **Fused 1-Pass** |

---

## 2. Architectural Design

```
                     +---------------------------------------+
                     | analysis.analyze_all() / SolverEngine |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | fast_scoring.score_matrix_and_bincounts|
                     +---------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
              (HAS_C_LIB = True)                  (HAS_C_LIB = False)
                       v                                   v
        +----------------------------+      +----------------------------+
        |  Multithreaded C Kernel    |      |  Optimized NumPy Fallback  |
        |  (_score_and_bincount_C)   |      |  (_score_matrix_numpy)     |
        +----------------------------+      +----------------------------+
```

### Auto-Compilation & Runtime Loading
- `fast_scoring.py` automatically compiles an inline POSIX-multithreaded C routine upon first load using system `clang` or `gcc`.
- The compiled shared object (`libscore_<hash>.dylib/.so/.dll`) is cached in `.wordle_cache/`.
- If no system C compiler is installed or compilation fails, `fast_scoring.py` sets `HAS_C_LIB = False` and falls back transparently to an optimized 5-step NumPy loop without raising errors.

---

## 3. Algorithm & Bitwise Mechanics

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

### Why Straight-Line C ALU Code Beats Bitmask Branching
Modern superscalar CPUs (Apple Silicon M-series & x86) feature deep execution pipelines with wide vector/ALU execution units. Branchless, unrolled C loops allow compilers (`clang -O3`) to emit straight-line SIMD instructions. Introducing bitmask presence filtering (`t->mask & (1U << g[i])`) adds conditional branches that incur branch misprediction penalties, yielding ~554M pairs/sec vs **987M pairs/sec** for straight-line ALU execution.

---

## 4. Usage & Benchmarks

### Running the Scoring Benchmark
```bash
uv run python benchmark_scoring.py
```

### Running Strategy Benchmarks
```bash
uv run python benchmark_strategies.py --sample-size 100
```

### Running Tests
```bash
uv run pytest
```
