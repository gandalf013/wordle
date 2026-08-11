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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

CACHE_DIR = Path(__file__).resolve().parent / ".wordle_cache"
CACHE_VERSION = 1

C_SOURCE_CODE = r"""
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <string.h>

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
"""


def _load_c_lib():
    """Attempt to compile and load the C scoring library. Return (lib, True) or (None, False)."""
    ext = ".dylib" if sys.platform == "darwin" else (".dll" if sys.platform == "win32" else ".so")
    src_hash = hashlib.sha256(C_SOURCE_CODE.encode()).hexdigest()[:12]
    lib_dir = CACHE_DIR
    lib_path = lib_dir / f"libscore_{src_hash}{ext}"

    if not lib_path.exists():
        compiler = shutil.which("clang") or shutil.which("gcc")
        if not compiler:
            return None, False
        lib_dir.mkdir(parents=True, exist_ok=True)
        c_src_path = lib_dir / f"score_{src_hash}.c"
        c_src_path.write_text(C_SOURCE_CODE)
        try:
            cmd = [compiler, "-O3", "-shared", "-fPIC", str(c_src_path), "-o", str(lib_path)]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
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
        return lib, True
    except Exception:
        return None, False


_C_LIB, HAS_C_LIB = _load_c_lib()
NUM_THREADS = min(os.cpu_count() or 4, 16)


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
            NUM_THREADS,
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
            NUM_THREADS,
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
