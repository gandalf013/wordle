"""Vectorized NumPy and C-accelerated scoring and on-disk caching for the (guess, target)
Wordle score matrix.

Game.get_score scores one (guess, target) pair at a time in pure Python,
which makes Game.get_all_censuses (score every guess against every
remaining target) the dominant cost of find_best_guess -- ~45s for the
full word list at round 1. score_matrix computes the same values for every
pair at once using multithreaded C routines (or batched NumPy array ops as
a fallback), and cached_score_matrix persists the result to disk so repeat
runs against an unchanged word list load instantly instead of recomputing.
"""

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from tqdm import tqdm

CACHE_DIR = Path(__file__).resolve().parent / ".wordle_cache"
CACHE_VERSION = 1

C_SOURCE_CODE = r"""
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

static inline uint8_t score_pair(const uint8_t* g, const uint8_t* t) {
    uint8_t gr0 = (g[0] == t[0]);
    uint8_t gr1 = (g[1] == t[1]);
    uint8_t gr2 = (g[2] == t[2]);
    uint8_t gr3 = (g[3] == t[3]);
    uint8_t gr4 = (g[4] == t[4]);

    int rem0 = (!gr0 && g[0] == t[0]) + (!gr1 && g[0] == t[1]) + (!gr2 && g[0] == t[2]) + (!gr3 && g[0] == t[3]) + (!gr4 && g[0] == t[4]);
    int rem1 = (!gr0 && g[1] == t[0]) + (!gr1 && g[1] == t[1]) + (!gr2 && g[1] == t[2]) + (!gr3 && g[1] == t[3]) + (!gr4 && g[1] == t[4]);
    int rem2 = (!gr0 && g[2] == t[0]) + (!gr1 && g[2] == t[1]) + (!gr2 && g[2] == t[2]) + (!gr3 && g[2] == t[3]) + (!gr4 && g[2] == t[4]);
    int rem3 = (!gr0 && g[3] == t[0]) + (!gr1 && g[3] == t[1]) + (!gr2 && g[3] == t[2]) + (!gr3 && g[3] == t[3]) + (!gr4 && g[3] == t[4]);
    int rem4 = (!gr0 && g[4] == t[0]) + (!gr1 && g[4] == t[1]) + (!gr2 && g[4] == t[2]) + (!gr3 && g[4] == t[3]) + (!gr4 && g[4] == t[4]);

    uint8_t y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0;
    if (!gr0 && rem0 > 0) { y0 = 1; rem0--; if (g[1] == g[0]) rem1--; if (g[2] == g[0]) rem2--; if (g[3] == g[0]) rem3--; if (g[4] == g[0]) rem4--; }
    if (!gr1 && rem1 > 0) { y1 = 1; rem1--; if (g[2] == g[1]) rem2--; if (g[3] == g[1]) rem3--; if (g[4] == g[1]) rem4--; }
    if (!gr2 && rem2 > 0) { y2 = 1; rem2--; if (g[3] == g[2]) rem3--; if (g[4] == g[2]) rem4--; }
    if (!gr3 && rem3 > 0) { y3 = 1; rem3--; if (g[4] == g[3]) rem4--; }
    if (!gr4 && rem4 > 0) { y4 = 1; }

    uint8_t s0 = gr0 ? 2 : y0;
    uint8_t s1 = gr1 ? 2 : y1;
    uint8_t s2 = gr2 ? 2 : y2;
    uint8_t s3 = gr3 ? 2 : y3;
    uint8_t s4 = gr4 ? 2 : y4;

    return s0 * 81 + s1 * 27 + s2 * 9 + s3 * 3 + s4;
}

typedef struct {
    const uint8_t* guesses;
    size_t g_start;
    size_t g_end;
    const uint8_t* targets;
    size_t T;
    const double* weights;
    uint8_t* out_matrix;
    double* out_counts;
    double* out_masses;
} fused_worker_args_t;

static void* fused_worker(void* arg) {
    fused_worker_args_t* a = (fused_worker_args_t*)arg;
    for (size_t g = a->g_start; g < a->g_end; g++) {
        const uint8_t* g_ptr = a->guesses + g * 5;
        double* row_counts = a->out_counts ? a->out_counts + g * 243 : NULL;
        double* row_masses = a->out_masses ? a->out_masses + g * 243 : NULL;
        uint8_t* row_matrix = a->out_matrix ? a->out_matrix + g * a->T : NULL;

        for (size_t t = 0; t < a->T; t++) {
            uint8_t s = score_pair(g_ptr, a->targets + t * 5);
            if (row_matrix) row_matrix[t] = s;
            if (row_counts) row_counts[s] += 1.0;
            if (row_masses) row_masses[s] += a->weights[t];
        }
    }
    return NULL;
}

void score_and_bincount_parallel(
    const uint8_t* guesses, size_t G,
    const uint8_t* targets, size_t T,
    const double* weights,
    uint8_t* out_matrix,
    double* out_counts,
    double* out_masses,
    int num_threads
) {
    if (out_counts) memset(out_counts, 0, G * 243 * sizeof(double));
    if (out_masses) memset(out_masses, 0, G * 243 * sizeof(double));

    pthread_t threads[num_threads];
    fused_worker_args_t args[num_threads];
    size_t chunk = (G + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        args[i].guesses = guesses;
        args[i].g_start = i * chunk;
        args[i].g_end = (i + 1) * chunk < G ? (i + 1) * chunk : G;
        args[i].targets = targets;
        args[i].T = T;
        args[i].weights = weights;
        args[i].out_matrix = out_matrix;
        args[i].out_counts = out_counts;
        args[i].out_masses = out_masses;
        pthread_create(&threads[i], NULL, fused_worker, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

/* score_bincount_stats_parallel additionally reduces each guess's 243-bucket
 * row (counts/masses) into the scalar summary stats analysis.analyze_all
 * needs (entropy, worst-case bucket, expected pool size, and weighted
 * variants) *in the same cache-resident pass* that just built the row, on
 * the same worker thread. The alternative -- handing the (G, 243) counts
 * array back to NumPy and reducing it there -- forces several extra
 * single-threaded full-array passes (one per np.where/log2/sum call), which
 * for realistic guess-list sizes costs more than the scoring itself. Writing
 * out_counts/out_masses is optional (only needed by callers that also want
 * TwoPlyExpectimaxStrategy's per-bucket detail); when NULL, the 243-wide row
 * lives only in the worker's stack and never touches G*243-sized memory. */
typedef struct {
    const uint8_t* guesses;
    size_t g_start;
    size_t g_end;
    const uint8_t* targets;
    size_t T;
    const double* weights;
    uint8_t* out_matrix;
    double* out_counts;
    double* out_masses;
    int want_stats;
    double* out_entropy;
    double* out_worst_case;
    double* out_expected_size;
    double* out_weighted_entropy;
    double* out_weighted_expected_size;
    double* out_total_mass;
} stats_worker_args_t;

static void* stats_worker(void* arg) {
    stats_worker_args_t* a = (stats_worker_args_t*)arg;
    double Tf = (double)a->T;
    int has_weights = a->weights != NULL;

    for (size_t g = a->g_start; g < a->g_end; g++) {
        const uint8_t* g_ptr = a->guesses + g * 5;
        uint8_t* row_matrix = a->out_matrix ? a->out_matrix + g * a->T : NULL;
        double row_counts[243];
        double row_masses[243];
        memset(row_counts, 0, sizeof(row_counts));
        if (has_weights) memset(row_masses, 0, sizeof(row_masses));

        for (size_t t = 0; t < a->T; t++) {
            uint8_t s = score_pair(g_ptr, a->targets + t * 5);
            if (row_matrix) row_matrix[t] = s;
            row_counts[s] += 1.0;
            if (has_weights) row_masses[s] += a->weights[t];
        }

        if (a->out_counts) memcpy(a->out_counts + g * 243, row_counts, sizeof(row_counts));
        if (a->out_masses && has_weights) memcpy(a->out_masses + g * 243, row_masses, sizeof(row_masses));

        if (a->want_stats) {
            double entropy = 0.0, worst = 0.0, expected = 0.0;
            for (int s = 0; s < 243; s++) {
                double c = row_counts[s];
                if (c > worst) worst = c;
                expected += c * c;
                if (c > 0.0) {
                    double p = c / Tf;
                    entropy -= p * log2(p);
                }
            }
            a->out_entropy[g] = entropy;
            a->out_worst_case[g] = worst;
            a->out_expected_size[g] = expected / Tf;

            if (has_weights) {
                double total = 0.0;
                for (int s = 0; s < 243; s++) total += row_masses[s];
                double wentropy = 0.0, wexpected = 0.0;
                if (total > 0.0) {
                    for (int s = 0; s < 243; s++) {
                        double m = row_masses[s];
                        if (m > 0.0) {
                            double wp = m / total;
                            wentropy -= wp * log2(wp);
                            wexpected += wp * row_counts[s];
                        }
                    }
                }
                a->out_weighted_entropy[g] = wentropy;
                a->out_weighted_expected_size[g] = wexpected;
                a->out_total_mass[g] = total;
            }
        }
    }
    return NULL;
}

void score_bincount_stats_parallel(
    const uint8_t* guesses, size_t G,
    const uint8_t* targets, size_t T,
    const double* weights,
    uint8_t* out_matrix,
    double* out_counts,
    double* out_masses,
    int want_stats,
    double* out_entropy,
    double* out_worst_case,
    double* out_expected_size,
    double* out_weighted_entropy,
    double* out_weighted_expected_size,
    double* out_total_mass,
    int num_threads
) {
    if (out_counts) memset(out_counts, 0, G * 243 * sizeof(double));
    if (out_masses) memset(out_masses, 0, G * 243 * sizeof(double));

    pthread_t threads[num_threads];
    stats_worker_args_t args[num_threads];
    size_t chunk = (G + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        args[i].guesses = guesses;
        args[i].g_start = i * chunk;
        args[i].g_end = (i + 1) * chunk < G ? (i + 1) * chunk : G;
        args[i].targets = targets;
        args[i].T = T;
        args[i].weights = weights;
        args[i].out_matrix = out_matrix;
        args[i].out_counts = out_counts;
        args[i].out_masses = out_masses;
        args[i].want_stats = want_stats;
        args[i].out_entropy = out_entropy;
        args[i].out_worst_case = out_worst_case;
        args[i].out_expected_size = out_expected_size;
        args[i].out_weighted_entropy = out_weighted_entropy;
        args[i].out_weighted_expected_size = out_weighted_expected_size;
        args[i].out_total_mass = out_total_mass;
        pthread_create(&threads[i], NULL, stats_worker, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}
"""


def _arch_tuning_flag() -> str | None:
    """The compiler flag that lets the vectorizer use this machine's full
    instruction set, appropriate to the *build* machine -- safe here only
    because the library is compiled at runtime on the same machine that
    will run it (see the cache-key comment below), never distributed as a
    prebuilt binary. x86_64's baseline ISA is SSE2-only; AVX2 (double the
    SIMD lanes) requires opting in via -march. ARM64's baseline already
    includes NEON, so there's no equivalent gap for -mcpu to close there,
    but passing it is harmless and keeps builds consistent across archs."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "i386", "i686"):
        return "-march=native"
    if machine in ("arm64", "aarch64"):
        return "-mcpu=native"
    return None


def _load_c_lib():
    """Attempt to compile and load the C scoring library. Return (lib, True) or (None, False)."""
    ext = ".dylib" if sys.platform == "darwin" else (".dll" if sys.platform == "win32" else ".so")

    # The compiled library is cached to disk keyed on a hash that includes
    # this machine's hostname and architecture, not just the C source --
    # otherwise a -march=native/-mcpu=native build (see _arch_tuning_flag)
    # copied to a different machine (e.g. an rsync'd home directory instead
    # of a fresh checkout) could reuse a stale binary tuned for a different
    # CPU and crash with an illegal instruction instead of recompiling.
    fingerprint = f"{platform.node()}|{platform.machine()}|{C_SOURCE_CODE}"
    src_hash = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    lib_dir = CACHE_DIR
    lib_path = lib_dir / f"libscore_{src_hash}{ext}"

    if not lib_path.exists():
        compiler = shutil.which("clang") or shutil.which("gcc")
        if not compiler:
            return None, False
        lib_dir.mkdir(parents=True, exist_ok=True)
        c_src_path = lib_dir / f"score_{src_hash}.c"
        c_src_path.write_text(C_SOURCE_CODE)

        # -pthread (not just -lm) is needed for correct pthread linking on
        # some libcs -- macOS links pthread symbols into libSystem
        # unconditionally so it's a no-op there, but musl (Alpine) and some
        # glibc configurations need it to actually resolve at link time.
        base_cmd = [compiler, "-O3", "-shared", "-fPIC", "-pthread"]
        arch_flag = _arch_tuning_flag()
        attempts = [base_cmd + [arch_flag]] if arch_flag else []
        attempts.append(base_cmd)

        compiled = False
        for extra_flags in attempts:
            cmd = [*extra_flags, str(c_src_path), "-o", str(lib_path), "-lm"]
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                compiled = True
                break
            except Exception:
                continue
        if not compiled:
            return None, False

    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.score_and_bincount_parallel.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        c_double_p = ctypes.POINTER(ctypes.c_double)
        lib.score_bincount_stats_parallel.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            c_double_p,
            ctypes.POINTER(ctypes.c_uint8),
            c_double_p,
            c_double_p,
            ctypes.c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            ctypes.c_int,
        ]
        return lib, True
    except Exception:
        return None, False


_C_LIB, HAS_C_LIB = _load_c_lib()
NUM_THREADS = min(os.cpu_count() or 4, 16)

# pthread_create/join has fixed per-thread overhead (tens of microseconds on
# this machine's heterogeneous P+E cores) that dominates wall-clock time for
# small jobs -- e.g. spawning 10 threads to score 15k guesses against a
# 10-word candidate pool costs more in thread setup than the ~150k pairs
# take to actually score. Scale thread count down for small jobs instead of
# always maxing out NUM_THREADS; empirically ~20k pairs/thread is where
# creation overhead stops dominating on this hardware.
_PAIRS_PER_THREAD = 20_000


def _thread_count_for(total_pairs: int) -> int:
    if total_pairs <= 0:
        return 1
    return max(1, min(NUM_THREADS, total_pairs // _PAIRS_PER_THREAD))


def words_to_codes(words: Sequence[str], n: int) -> np.ndarray:
    """(len(words), n) uint8 array of 0-25 letter codes; every word must be
    lowercase ASCII of length n."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    return (arr.reshape(len(words), n) - ord("a")).astype(np.uint8)


def _score_batch_numpy(guess_batch, target_codes, target_counts, place_values, n):
    bG = guess_batch.shape[0]
    T = target_codes.shape[0]

    green = guess_batch[:, None, :] == target_codes[None, :, :]  # (bG, T, n)
    same_letter = guess_batch[:, :, None] == guess_batch[:, None, :]  # (bG, n, n)

    gathered = target_counts[:, guess_batch].transpose(1, 0, 2)  # (bG, T, n)

    green_same_letter = np.zeros((bG, T, n), dtype=np.int16)
    for i in range(n):
        for j in range(n):
            mask = same_letter[:, i, j]
            if np.any(mask):
                green_same_letter[mask, :, i] += green[mask, :, j]

    remaining = gathered - green_same_letter

    yellow = np.zeros((bG, T, n), dtype=bool)
    for i in range(n):
        eligible = (~green[:, :, i]) & (remaining[:, :, i] > 0)
        yellow[:, :, i] = eligible
        sub = eligible[:, :, None] * same_letter[:, i, :][:, None, :]
        remaining -= sub

    category = (green.astype(np.uint8) * np.uint8(2)) | yellow.astype(np.uint8)
    return (category * place_values).sum(axis=2, dtype=np.uint8)


def _score_matrix_c(
    guesses: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 1000,
    show_progress: bool | None = None,
) -> np.ndarray:
    n = len(targets[0])
    G, T = len(guesses), len(targets)
    guess_codes = words_to_codes(guesses, n)
    target_codes = words_to_codes(targets, n)
    out = np.empty((G, T), dtype=np.uint8)

    should_show = sys.stderr.isatty() if show_progress is None else show_progress
    batches = range(0, G, batch_size)
    if should_show and G > batch_size:
        batches = tqdm(batches, total=-(-G // batch_size))

    for start in batches:
        end = min(start + batch_size, G)
        batch_G = end - start
        _C_LIB.score_and_bincount_parallel(
            guess_codes[start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            batch_G,
            target_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            T,
            None,
            out[start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None,
            None,
            _thread_count_for(batch_G * T),
        )
    return out


def _score_matrix_numpy(
    guesses: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 1000,
    show_progress: bool | None = None,
) -> np.ndarray:
    n = len(targets[0])
    G, T = len(guesses), len(targets)
    guess_codes = words_to_codes(guesses, n)
    target_codes = words_to_codes(targets, n)

    target_counts = np.zeros((T, 26), dtype=np.int16)
    for i in range(n):
        np.add.at(target_counts, (np.arange(T), target_codes[:, i]), 1)

    place_values = (3 ** np.arange(n - 1, -1, -1)).astype(np.uint8)

    out = np.empty((G, T), dtype=np.uint8)
    batches = range(0, G, batch_size)
    should_show = sys.stderr.isatty() if show_progress is None else show_progress
    if should_show and G > batch_size:
        batches = tqdm(batches, total=-(-G // batch_size))
    for start in batches:
        end = start + batch_size
        out[start:end] = _score_batch_numpy(
            guess_codes[start:end], target_codes, target_counts, place_values, n
        )
    return out


def score_matrix(
    guesses: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 1000,
    show_progress: bool | None = None,
) -> np.ndarray:
    """Packed base-3 score of every guess against every target, as a
    (len(guesses), len(targets)) uint8 array, where matrix[g, t] equals
    Game.get_score(guesses[g], targets[t])."""
    if HAS_C_LIB and len(guesses) > 0 and len(targets) > 0 and len(targets[0]) == 5:
        return _score_matrix_c(guesses, targets, batch_size=batch_size, show_progress=show_progress)
    return _score_matrix_numpy(guesses, targets, batch_size=batch_size, show_progress=show_progress)


def score_matrix_and_bincounts(
    guesses: Sequence[str],
    targets: Sequence[str],
    weights: dict[str, float] | None = None,
    use_cache: bool = False,
    show_progress: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute score matrix, unweighted bucket counts, and weighted bucket masses
    in a single pass (using fused multithreaded C routine if available, or falling back
    to score_matrix + bincount_scores)."""
    if use_cache:
        matrix = cached_score_matrix(guesses, targets, show_progress=show_progress)
        target_weights = (
            np.array([weights.get(w, 1.0) for w in targets], dtype=np.float64)
            if weights is not None
            else None
        )
        counts, masses = bincount_scores(matrix, weights=target_weights)
        return matrix, counts, masses

    n = len(targets[0]) if targets else 5
    G, T = len(guesses), len(targets)
    if not G or not T:
        empty_mat = np.empty((G, T), dtype=np.uint8)
        empty_cnt = np.empty((G, 243), dtype=np.float64)
        return empty_mat, empty_cnt, empty_cnt

    if HAS_C_LIB and n == 5:
        guess_codes = words_to_codes(guesses, n)
        target_codes = words_to_codes(targets, n)
        target_weights = (
            np.array([weights.get(w, 1.0) for w in targets], dtype=np.float64)
            if weights is not None
            else None
        )

        matrix = np.empty((G, T), dtype=np.uint8)
        counts = np.empty((G, 243), dtype=np.float64)
        masses = counts if target_weights is None else np.empty((G, 243), dtype=np.float64)

        _C_LIB.score_and_bincount_parallel(
            guess_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            G,
            target_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            T,
            target_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if target_weights is not None
            else None,
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if target_weights is not None
            else None,
            _thread_count_for(G * T),
        )
        return matrix, counts, masses

    matrix = score_matrix(guesses, targets, show_progress=show_progress)
    target_weights = (
        np.array([weights.get(w, 1.0) for w in targets], dtype=np.float64)
        if weights is not None
        else None
    )
    counts, masses = bincount_scores(matrix, weights=target_weights)
    return matrix, counts, masses


class ScoringStats(NamedTuple):
    """Everything analysis.analyze_all needs per guess, computed in one pass.

    `matrix`/`counts`/`masses` are None unless requested via
    `need_matrix`/`need_bucket_arrays` -- most callers only rank guesses by
    the scalar fields, so score_and_analyze skips materializing (G, T) or
    (G, 243) arrays for them. `weighted_entropy`/`weighted_expected_size`/
    `total_mass` are None when `weights` wasn't supplied.
    """

    matrix: np.ndarray | None
    counts: np.ndarray | None
    masses: np.ndarray | None
    entropy: np.ndarray
    worst_case_size: np.ndarray
    expected_size: np.ndarray
    weighted_entropy: np.ndarray | None
    weighted_expected_size: np.ndarray | None
    total_mass: np.ndarray | None


def _stats_from_counts_masses(counts, masses, T, weights_present):
    """NumPy fallback reduction, used when the C kernel isn't available.
    Mirrors stats_worker's math exactly (same formulas, just batched over
    the full (G, 243) array instead of fused into the per-guess C loop)."""
    probs = counts / T
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -np.sum(probs * log_probs, axis=1)
    worst_case_size = np.max(counts, axis=1)
    expected_size = np.sum(counts**2, axis=1) / T

    if not weights_present:
        return entropy, worst_case_size, expected_size, None, None, None

    total_mass = np.sum(masses, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_probs = np.where(total_mass[:, None] > 0, masses / total_mass[:, None], 0.0)
        w_log_probs = np.where(w_probs > 0, np.log2(w_probs), 0.0)
        weighted_entropy = -np.sum(w_probs * w_log_probs, axis=1)
    weighted_expected_size = np.sum(w_probs * counts, axis=1)
    return entropy, worst_case_size, expected_size, weighted_entropy, weighted_expected_size, total_mass


def score_and_analyze(
    guesses: Sequence[str],
    targets: Sequence[str],
    weights: dict[str, float] | None = None,
    need_matrix: bool = False,
    need_bucket_arrays: bool = False,
    use_cache: bool = False,
    show_progress: bool | None = None,
) -> ScoringStats:
    """Score every guess against every target and reduce straight to the
    per-guess summary stats analysis.analyze_all needs (entropy, worst-case
    bucket, expected pool size, and weighted variants), without ever
    materializing a (G, 243) bucket-count array unless `need_bucket_arrays`
    is set.

    Doing the entropy/expected-size reduction as a NumPy pass over
    (G, 243) counts -- as analyze_all used to -- costs more wall-clock time
    than scoring itself once G is a full guess list (~3-15k words): each
    np.where/log2/sum call is a separate single-threaded pass over a
    multi-megabyte array. Folding the reduction into the same multithreaded
    C loop that just built each guess's 243-bucket row (still hot in that
    worker thread's cache) avoids that entirely.
    """
    n = len(targets[0]) if targets else 5
    G, T = len(guesses), len(targets)

    if not G or not T:
        empty_mat = np.empty((G, T), dtype=np.uint8) if need_matrix else None
        empty_bucket = np.empty((G, 243), dtype=np.float64) if need_bucket_arrays else None
        empty_g = np.empty(G, dtype=np.float64)
        has_w = weights is not None
        return ScoringStats(
            matrix=empty_mat,
            counts=empty_bucket,
            masses=empty_bucket,
            entropy=empty_g,
            worst_case_size=empty_g,
            expected_size=empty_g,
            weighted_entropy=empty_g if has_w else None,
            weighted_expected_size=empty_g if has_w else None,
            total_mass=empty_g if has_w else None,
        )

    if use_cache:
        # The disk cache stores the raw score matrix, not the reduced stats,
        # so a cache hit still needs a reduction pass -- but skips the
        # scoring pass entirely (the expensive part on a cache miss, e.g. an
        # uncached round 1 against the full word list). Reduction goes
        # through the same NumPy path as the C-unavailable fallback below;
        # use_cache is only set for round 1 (SolverEngine.get_analyses),
        # which happens once per process, so it isn't worth a second C
        # entry point that reduces an already-built matrix instead of
        # scoring one.
        matrix, counts, masses = score_matrix_and_bincounts(
            guesses, targets, weights=weights, use_cache=True, show_progress=show_progress
        )
        (
            entropy,
            worst_case_size,
            expected_size,
            weighted_entropy,
            weighted_expected_size,
            total_mass,
        ) = _stats_from_counts_masses(counts, masses, T, weights is not None)
        return ScoringStats(
            matrix=matrix if need_matrix else None,
            counts=counts if need_bucket_arrays else None,
            masses=masses if need_bucket_arrays else None,
            entropy=entropy,
            worst_case_size=worst_case_size,
            expected_size=expected_size,
            weighted_entropy=weighted_entropy,
            weighted_expected_size=weighted_expected_size,
            total_mass=total_mass,
        )

    if HAS_C_LIB and G and T and n == 5:
        guess_codes = words_to_codes(guesses, n)
        target_codes = words_to_codes(targets, n)
        target_weights = (
            np.array([weights.get(w, 1.0) for w in targets], dtype=np.float64)
            if weights is not None
            else None
        )

        matrix = np.empty((G, T), dtype=np.uint8) if need_matrix else None
        counts = np.empty((G, 243), dtype=np.float64) if need_bucket_arrays else None
        masses = (
            (counts if target_weights is None else np.empty((G, 243), dtype=np.float64))
            if need_bucket_arrays
            else None
        )
        entropy = np.empty(G, dtype=np.float64)
        worst_case_size = np.empty(G, dtype=np.float64)
        expected_size = np.empty(G, dtype=np.float64)
        weighted_entropy = np.empty(G, dtype=np.float64) if target_weights is not None else None
        weighted_expected_size = np.empty(G, dtype=np.float64) if target_weights is not None else None
        total_mass = np.empty(G, dtype=np.float64) if target_weights is not None else None

        c_double_p = ctypes.POINTER(ctypes.c_double)

        def as_double_p(arr):
            return None if arr is None else arr.ctypes.data_as(c_double_p)

        _C_LIB.score_bincount_stats_parallel(
            guess_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            G,
            target_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            T,
            as_double_p(target_weights),
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if matrix is not None else None,
            as_double_p(counts),
            as_double_p(masses),
            1,
            as_double_p(entropy),
            as_double_p(worst_case_size),
            as_double_p(expected_size),
            as_double_p(weighted_entropy),
            as_double_p(weighted_expected_size),
            as_double_p(total_mass),
            _thread_count_for(G * T),
        )
        return ScoringStats(
            matrix=matrix,
            counts=counts,
            masses=masses,
            entropy=entropy,
            worst_case_size=worst_case_size,
            expected_size=expected_size,
            weighted_entropy=weighted_entropy,
            weighted_expected_size=weighted_expected_size,
            total_mass=total_mass,
        )

    matrix, counts, masses = score_matrix_and_bincounts(
        guesses, targets, weights=weights, show_progress=show_progress
    )
    (
        entropy,
        worst_case_size,
        expected_size,
        weighted_entropy,
        weighted_expected_size,
        total_mass,
    ) = _stats_from_counts_masses(counts, masses, T, weights is not None)
    return ScoringStats(
        matrix=matrix if need_matrix else None,
        counts=counts if need_bucket_arrays else None,
        masses=masses if need_bucket_arrays else None,
        entropy=entropy,
        worst_case_size=worst_case_size,
        expected_size=expected_size,
        weighted_entropy=weighted_entropy,
        weighted_expected_size=weighted_expected_size,
        total_mass=total_mass,
    )


def bincount_scores(scores, weights=None, minlength=243):
    """Bucket a packed base-3 score array (from score_matrix) by value.

    `scores` is 1-D (one guess's row) or 2-D (stacked rows, one per guess).
    Returns `(counts, masses)`, each shaped like `scores` but with a
    trailing `minlength`-sized bucket axis: `counts` is the raw per-bucket
    tally, and `masses` is the same tally weighted by `weights` (aligned to
    scores' target axis) -- or `counts` itself when `weights` is None, so
    callers can use `masses` unconditionally instead of branching."""
    if scores.ndim == 1:
        counts = np.bincount(scores, minlength=minlength).astype(np.float64)
        if weights is None:
            return counts, counts
        masses = np.bincount(scores, weights=weights, minlength=minlength)
        return counts, masses

    counts = np.array([np.bincount(row, minlength=minlength) for row in scores], dtype=np.float64)
    if weights is None:
        return counts, counts
    masses = np.array(
        [np.bincount(row, weights=weights, minlength=minlength) for row in scores],
        dtype=np.float64,
    )
    return counts, masses


def cache_key(guesses, targets):
    h = hashlib.sha256()
    h.update("\n".join(guesses).encode())
    h.update(b"\x00")
    h.update("\n".join(targets).encode())
    h.update(f"\x00v{CACHE_VERSION}".encode())
    return h.hexdigest()[:24]


def cached_score_matrix(
    guesses: Sequence[str],
    targets: Sequence[str],
    cache_dir=None,
    batch_size: int = 1000,
    show_progress: bool | None = None,
) -> np.ndarray:
    """score_matrix(guesses, targets), backed by an on-disk cache keyed on
    the exact ordered word lists (+ CACHE_VERSION). A hit is a single
    np.load; a miss computes and persists the matrix for next time."""
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    path = cache_dir / f"{cache_key(guesses, targets)}.npy"
    if path.exists():
        return np.load(path)

    matrix = score_matrix(
        guesses, targets, batch_size=batch_size, show_progress=show_progress
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, matrix)
    tmp.rename(path)
    return matrix
