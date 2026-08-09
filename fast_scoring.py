"""Vectorized NumPy scoring and on-disk caching for the (guess, target)
Wordle score matrix.

Game.get_score scores one (guess, target) pair at a time in pure Python,
which makes Game.get_all_censuses (score every guess against every
remaining target) the dominant cost of find_best_guess -- ~45s for the
full word list at round 1. score_matrix computes the same values for every
pair at once using batched NumPy array ops, and cached_score_matrix
persists the result to disk so repeat runs against an unchanged word list
(the common case: round 1 against the full target list) load instantly
instead of recomputing.

score_matrix mirrors Game.get_score's two-pass algorithm exactly (green
pass consumes letter counts first, then a left-to-right yellow pass against
what's left) but batches it:

  - green[g, t, i]: guess[g, i] == target[t, i] -- a direct elementwise
    comparison, no counting needed.
  - remaining[g, t, i]: how much of guess[g, i]'s letter is left in target
    t's counts after every green match of that letter (anywhere in the
    word) is subtracted. Positions sharing a letter within one guess word
    share this same value, since it only depends on the letter identity.
  - The yellow pass then runs sequentially over the (small, fixed) n
    positions, marking a position yellow if it isn't green and its shared
    letter-pool is still positive, then decrementing that pool everywhere
    the letter recurs in the guess -- exactly mirroring the scalar
    left-to-right Counter decrement, but for every (guess, target) pair at
    once.
"""

import hashlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

CACHE_DIR = Path(__file__).resolve().parent / ".wordle_cache"
CACHE_VERSION = 1


def words_to_codes(words: Sequence[str], n: int) -> np.ndarray:
    """(len(words), n) uint8 array of 0-25 letter codes; every word must be
    lowercase ASCII of length n."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    return (arr.reshape(len(words), n) - ord("a")).astype(np.uint8)


def _score_batch(guess_batch, target_codes, target_counts, place_values, n):
    bG = guess_batch.shape[0]
    T = target_codes.shape[0]

    green = guess_batch[:, None, :] == target_codes[None, :, :]  # (bG, T, n)
    same_letter = guess_batch[:, :, None] == guess_batch[:, None, :]  # (bG, n, n)

    gathered = target_counts[:, guess_batch].transpose(1, 0, 2)  # (bG, T, n)
    green_same_letter = np.einsum(
        "gtj,gij->gti", green.astype(np.int16), same_letter.astype(np.int16)
    )
    remaining = (gathered - green_same_letter).astype(np.int16)

    yellow = np.zeros((bG, T, n), dtype=bool)
    for i in range(n):
        eligible = (~green[:, :, i]) & (remaining[:, :, i] > 0)
        yellow[:, :, i] = eligible
        remaining -= eligible[:, :, None] * same_letter[:, i, :][:, None, :]

    category = green.astype(np.uint8) * np.uint8(2) + yellow.astype(np.uint8)
    return (category * place_values[None, None, :]).sum(axis=2, dtype=np.uint8)


def score_matrix(
    guesses: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 1000,
    show_progress: bool | None = None,
) -> np.ndarray:
    """Packed base-3 score of every guess against every target, as a
    (len(guesses), len(targets)) uint8 array, where matrix[g, t] equals
    Game.get_score(guesses[g], targets[t])."""
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
        out[start:end] = _score_batch(
            guess_codes[start:end], target_codes, target_counts, place_values, n
        )
    return out


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
