"""All output formatting: turning a packed score, a GuessAnalysis, or a
guess/score history into strings for a terminal. cli.py is the only caller
-- every function here is pure (str in, str out), so it's testable without
mocking input()/print() the way cli.py's own tests have to.
"""

from analysis import GuessAnalysis
from scoring import get_score_list

SQUARES = {0: "⬛", 1: "🟨", 2: "🟩"}


def format_score(score: int, n: int) -> str:
    """Emoji rendering of a packed score, e.g. '⬛🟨🟩⬛🟩'."""
    return "".join(SQUARES[s] for s in get_score_list(score, n))


def format_history(history: list[tuple[str, int]], n: int) -> str:
    """The 'guess EMOJI_ROW' lines shown when a game is solved."""
    return "\n".join(f"{guess} {format_score(score, n)}" for guess, score in history)


def format_top_guesses(
    analyses: list[GuessAnalysis], top_n: int = 10, weighted: bool = False
) -> str:
    """Table of the top-n analyses, in the order given -- callers pass
    already-ranked analyses (e.g. from Strategy.rank()). When weighted=True
    and an analysis carries weighted fields, adds P(answer) and
    weighted-entropy columns alongside the uniform ones, so both views are
    visible at once rather than losing the raw-count numbers. An analysis
    without weighted fields (weights weren't supplied to analyze_all) shows
    "-" in those columns rather than being dropped.
    """
    header = ["guess", "entropy", "expected", "worst"]
    if weighted:
        header += ["w.entropy", "w.expected", "P(answer)"]

    rows = [header]
    for a in analyses[:top_n]:
        row = [a.guess, f"{a.entropy:.4f}", f"{a.expected_size:.2f}", str(a.worst_case_size)]
        if weighted:
            row.append(f"{a.weighted_entropy:.4f}" if a.weighted_entropy is not None else "-")
            row.append(
                f"{a.weighted_expected_size:.2f}"
                if a.weighted_expected_size is not None
                else "-"
            )
            row.append(
                f"{a.solution_probability:.4f}"
                if a.solution_probability is not None
                else "-"
            )
        rows.append(row)

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows
    )


def format_buckets(
    analysis: GuessAnalysis,
    limit: int | None = None,
    weights: dict[str, float] | None = None,
) -> str:
    """Score pattern -> bucket size (+ up to 5 sample words), largest
    buckets first. `weights` isn't carried on GuessAnalysis itself (only
    the already-aggregated weighted_* summary fields are) -- pass the same
    weights dict used to build `analysis` to sort/annotate by bucket weight
    mass instead of raw word count.
    """
    if analysis.buckets is None:
        return "<No bucket details available for this analysis>"

    n = len(analysis.guess)
    items = list(analysis.buckets.items())

    masses: dict[int, float] | None = None
    if weights is not None:
        masses = {s: sum(weights.get(w, 1.0) for w in words) for s, words in items}
        items.sort(key=lambda kv: masses[kv[0]], reverse=True)
    else:
        items.sort(key=lambda kv: len(kv[1]), reverse=True)

    if limit is not None:
        items = items[:limit]

    lines = []
    for score, words in items:
        sample = ", ".join(sorted(words)[:5])
        if len(words) > 5:
            sample += ", ..."
        count_desc = f"{len(words)} words"
        if masses is not None:
            count_desc += f" (mass {masses[score]:.3f})"
        lines.append(f"{format_score(score, n)}  {count_desc}  {sample}")

    return "\n".join(lines)
