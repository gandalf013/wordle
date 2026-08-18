"""Pure Wordle scoring: the scalar (guess, target) -> packed score
algorithm, plus encode/decode between the packed base-3 int and a
per-letter Score list.

Split out of the old Game class so nothing that only needs to score a
single pair (engine.py's candidate narrowing, tests, ad-hoc scripts) has to
carry Game's I/O and state machinery along with it. fast_scoring.py is the
vectorized batch version of the same algorithm, used when scoring many
pairs at once. Rendering a score for display (emoji squares) is
display.py's job, not this module's -- scoring.py only knows about the
packed int and its decoded Score list.
"""

import collections
from enum import IntEnum


class Score(IntEnum):
    GRAY = 0
    YELLOW = 1
    GREEN = 2


def get_score(guess: str, target: str) -> int:
    """Packed base-3 score of `guess` against `target`."""
    if len(guess) != len(target):
        raise ValueError(f"Guess {guess} not valid for target {target}")

    n = len(guess)
    c = collections.Counter(target)
    score = [Score.GRAY] * n
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            score[i] = Score.GREEN
            c[g] -= 1

    for i, g in enumerate(guess):
        if score[i] != Score.GREEN and c[g]:
            score[i] = Score.YELLOW
            c[g] -= 1

    return get_score_num(score)


def get_score_num(score) -> int:
    n = len(score)
    return sum(3 ** (n - i - 1) * s for i, s in enumerate(score))


def get_score_list(score: int, n: int) -> list[Score]:
    if score < 0 or score >= 3**n:
        raise ValueError(f"Score {score} out of bounds for word length {n}")
    r = []
    while score:
        score, rem = divmod(score, 3)
        r.append(Score(rem))

    r.extend([Score.GRAY] * (n - len(r)))
    return r[::-1]
