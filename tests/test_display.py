"""Tests for display.py's formatting functions. All pure str-in/str-out,
so no input()/print() mocking needed -- that's cli.py's job, tested in
test_cli.py.
"""

import analysis
from analysis import GuessAnalysis
from display import format_buckets, format_history, format_pool, format_score, format_top_guesses
from scoring import Score, get_score_num
from strategies import EntropyStrategy


def _analysis(guess, **kwargs):
    defaults = dict(
        buckets={},
        entropy=0.0,
        worst_case_size=0,
        expected_size=0.0,
        is_possible_solution=False,
    )
    defaults.update(kwargs)
    return GuessAnalysis(guess=guess, **defaults)


class TestFormatScore:
    def test_renders_expected_emoji(self):
        num = get_score_num(
            [Score.GRAY, Score.YELLOW, Score.GREEN, Score.GRAY, Score.GREEN]
        )
        assert format_score(num, 5) == "⬛🟨🟩⬛🟩"

    def test_all_gray(self):
        assert format_score(0, 2) == "⬛⬛"

    def test_all_green(self):
        assert format_score(get_score_num([Score.GREEN] * 2), 2) == "🟩🟩"


class TestFormatHistory:
    def test_renders_one_line_per_entry(self):
        history = [("aa", 0), ("bb", get_score_num([Score.GREEN] * 2))]
        assert format_history(history, 2) == "aa ⬛⬛\nbb 🟩🟩"

    def test_empty_history_is_empty_string(self):
        assert format_history([], 2) == ""


class TestFormatTopGuesses:
    def test_ranked_guess_appears_first(self):
        results = analysis.analyze_all(["ax", "aa"], ["aa", "ab", "ac"])
        ranked = EntropyStrategy().rank(results)
        table = format_top_guesses(ranked, top_n=2)
        lines = table.splitlines()
        assert lines[0].startswith("guess")
        assert lines[1].startswith("aa")

    def test_top_n_limits_rows(self):
        results = analysis.analyze_all(["ax", "aa"], ["aa", "ab", "ac"])
        table = format_top_guesses(results, top_n=1)
        # header + exactly one data row
        assert len(table.splitlines()) == 2

    def test_weighted_adds_extra_columns(self):
        weighted = _analysis(
            "aa", weighted_entropy=1.0, weighted_expected_size=2.0, solution_probability=0.5
        )
        table = format_top_guesses([weighted], weighted=True)
        header, row = table.splitlines()
        assert "w.entropy" in header
        assert "P(answer)" in header
        assert "0.5000" in row

    def test_weighted_falls_back_to_dash_when_analysis_has_no_weighted_fields(self):
        unweighted = _analysis("zz")
        table = format_top_guesses([unweighted], weighted=True)
        _, row = table.splitlines()
        assert "-" in row.split()


class TestFormatPool:
    def test_no_weights_sorts_alphabetically(self):
        lines = format_pool(["cc", "aa", "bb"]).splitlines()
        assert lines[:-1] == ["aa", "bb", "cc"]

    def test_weights_sort_most_likely_first(self):
        weights = {"aa": 1.0, "bb": 5.0, "cc": 2.0}
        lines = format_pool(["aa", "bb", "cc"], weights=weights).splitlines()
        assert [line.split()[0] for line in lines[:-1]] == ["bb", "cc", "aa"]

    def test_equal_weights_break_ties_alphabetically(self):
        weights = {"cc": 1.0, "aa": 1.0, "bb": 1.0}
        lines = format_pool(["cc", "aa", "bb"], weights=weights).splitlines()
        assert [line.split()[0] for line in lines[:-1]] == ["aa", "bb", "cc"]

    def test_missing_weight_defaults_to_one(self):
        lines = format_pool(["aa", "bb"], weights={"bb": 5.0}).splitlines()
        assert [line.split()[0] for line in lines[:-1]] == ["bb", "aa"]

    def test_trailing_summary_line_has_total_count(self):
        lines = format_pool(["aa", "bb", "cc"]).splitlines()
        assert lines[-1] == "(3 words)"

    def test_singular_word_count(self):
        lines = format_pool(["aa"]).splitlines()
        assert lines[-1] == "(1 word)"

    def test_limit_truncates_and_notes_omitted_count(self):
        lines = format_pool(["cc", "aa", "bb"], limit=2).splitlines()
        assert lines == ["aa", "bb", "(2 of 3 words shown)"]

    def test_limit_larger_than_pool_shows_all_without_truncation_note(self):
        lines = format_pool(["aa", "bb"], limit=10).splitlines()
        assert lines[-1] == "(2 words)"


class TestFormatBuckets:
    def test_largest_bucket_first(self):
        result = analysis.analyze("aa", ["aa", "ab", "ac"])
        lines = format_buckets(result).splitlines()
        # "aa" splits ["aa","ab","ac"] into a 1-word bucket and a 2-word
        # bucket -- the 2-word one should be listed first.
        assert "2 words" in lines[0]
        assert "1 words" in lines[1]

    def test_limit_truncates_output(self):
        result = analysis.analyze("aa", ["aa", "ab", "ac"])
        assert len(format_buckets(result, limit=1).splitlines()) == 1

    def test_sample_words_are_sorted_and_shown(self):
        result = analysis.analyze("ax", ["ac", "aa", "ab"])
        line = format_buckets(result).splitlines()[0]
        assert "aa, ab, ac" in line

    def test_weighted_sorts_by_mass_and_annotates_it(self):
        # "bb" splits ["aa","ab","ca","cb","cc","cd"] into a 4-word bucket
        # and a 2-word bucket; with these weights the 2-word bucket (which
        # contains the high-weight "ab") has more mass than the 4-word one.
        targets = ["aa", "ab", "ca", "cb", "cc", "cd"]
        weights = {"aa": 1.0, "ab": 100.0, "ca": 1.0, "cb": 1.0, "cc": 1.0, "cd": 1.0}
        result = analysis.analyze("bb", targets, weights=weights)
        lines = format_buckets(result, weights=weights).splitlines()
        assert "mass" in lines[0]
        assert "2 words" in lines[0]

    def test_handles_none_buckets(self):
        a = GuessAnalysis(
            guess="test",
            entropy=1.0,
            worst_case_size=1,
            expected_size=1.0,
            is_possible_solution=True,
            buckets=None,
        )
        assert "No bucket details" in format_buckets(a)
