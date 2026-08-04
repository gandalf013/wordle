"""Word list parsing.

Split out of wordle.py so callers that only need parsed word lists (e.g. a
future analysis/strategy layer) don't have to import the interactive Game
machinery to get them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WordList:
    """Parsed word list. `weights` covers every word in target+extra; words
    with no explicit weight column default to 1.0 (uniform), so
    words.wordle.txt (no weights) and words.weighted.txt (weighted) are
    interchangeable inputs downstream -- callers never need to branch on
    which kind of file they got.
    """

    target: list[str]
    extra: list[str]
    word_length: int
    weights: dict[str, float]


def parse_file(fp) -> WordList:
    """Parse target words, a blank line, then extra guess-only words.
    Each line is 'word' or 'word <weight>'.
    """
    target = []
    extra = []
    r = target
    wordlen = None
    weights = {}
    for line in fp:
        data = line.strip()
        if not data:
            if extra:
                raise ValueError("too many blank lines")
            r = extra
            continue

        parts = data.split()
        word = parts[0]
        if wordlen is None:
            wordlen = len(word)
        elif len(word) != wordlen:
            raise ValueError(f"Bad length {len(word)}, expected {wordlen}")

        r.append(word)
        weights[word] = float(parts[1]) if len(parts) > 1 else 1.0

    if set(target) & set(extra):
        raise ValueError("Target and extra words overlap")

    return WordList(target=target, extra=extra, word_length=wordlen, weights=weights)
