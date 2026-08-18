"""Tests for strategies.py.

EntropyStrategy(weighted=False)'s expected winners below (including the
real-word-list golden value "tarse") were originally cross-checked live
against the old Game.find_best_guess before Game was retired in favor of
engine.SolverEngine. They're asserted directly here now rather than
re-deriving them from a live comparison each run. ExpectedPoolSizeStrategy
and MinimaxStrategy have no Game equivalent to cross-check against (they're
new), so their tests verify the ranking directly against hand-computed
expectations.
"""

from pathlib import Path

import pytest

import analysis
import scoring
from analysis import GuessAnalysis
from engine import RoundOutcome, SolverEngine
from strategies import (
    BaseStrategy,
    EntropyStrategy,
    ExpectedPoolSizeStrategy,
    MaxBinsBalanceStrategy,
    MinimaxStrategy,
    NumBinsStrategy,
    TwoPlyExpectimaxStrategy,
)
from wordlists import parse_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def _analysis(
    guess,
    *,
    entropy=0.0,
    worst_case_size=0,
    expected_size=0.0,
    is_possible_solution=False,
    weighted_entropy=None,
    weighted_expected_size=None,
    solution_probability=None,
    bucket_counts=None,
    bucket_masses=None,
):
    """Build a GuessAnalysis directly rather than deriving it from real word
    scoring, for tests that need specific (e.g. exactly-tied) field values
    that are otherwise impractical to hit reliably via real buckets."""
    return GuessAnalysis(
        guess=guess,
        buckets={},
        entropy=entropy,
        worst_case_size=worst_case_size,
        expected_size=expected_size,
        is_possible_solution=is_possible_solution,
        weighted_entropy=weighted_entropy,
        weighted_expected_size=weighted_expected_size,
        solution_probability=solution_probability,
        bucket_counts=bucket_counts,
        bucket_masses=bucket_masses,
    )


class TestBaseStrategy:
    def test_empty_analyses_returns_empty_list(self):
        class DummyStrategy(BaseStrategy):
            def _rank(self, analyses, guesses_remaining=None):
                raise AssertionError("_rank should not be called when analyses is empty")

        assert DummyStrategy().rank([]) == []

    def test_non_empty_analyses_delegates_to_subclass(self):
        class DummyStrategy(BaseStrategy):
            def _rank(self, analyses, guesses_remaining=None):
                return list(reversed(analyses))

        a1 = _analysis("a1")
        a2 = _analysis("a2")
        assert DummyStrategy().rank([a1, a2]) == [a2, a1]

    def test_unimplemented_rank_raises(self):
        with pytest.raises(NotImplementedError):
            BaseStrategy().rank([_analysis("a")])

    @pytest.mark.parametrize(
        "strategy_cls",
        [
            EntropyStrategy,
            ExpectedPoolSizeStrategy,
            NumBinsStrategy,
            MaxBinsBalanceStrategy,
            MinimaxStrategy,
            TwoPlyExpectimaxStrategy,
        ],
    )
    def test_all_concrete_strategies_handle_empty_analyses(self, strategy_cls):
        assert strategy_cls().rank([]) == []


class TestEntropyStrategyMatchesGame:
    def test_picks_the_higher_entropy_guess(self):
        guesses, targets = ["ax", "aa"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets)
        ranked = EntropyStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_tie_break_prefers_a_possible_solution(self):
        # "aa" and "bb" have identical entropy against these targets; only
        # "aa" is an actual candidate solution.
        guesses, targets = ["bb", "aa"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets)
        ranked = EntropyStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_rank_returns_all_analyses_best_first(self):
        guesses, targets = ["ax", "aa", "bb"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets)
        ranked = EntropyStrategy().rank(results)
        assert {r.guess for r in ranked} == {r.guess for r in results}
        assert len(ranked) == len(results)
        assert ranked[0].guess == "aa"

    def test_zero_entropy_tie_break_with_abs_tol(self):
        # Two non-candidate / candidate guesses near 0.0 entropy (e.g. 0.0 and 1e-15).
        # Ensures math.isclose(..., abs_tol=1e-9) treats them as tied and favors
        # the possible solution.
        non_candidate = _analysis("nc", entropy=0.0, is_possible_solution=False)
        candidate = _analysis(
            "cs", entropy=1e-15, is_possible_solution=True, solution_probability=0.5
        )
        ranked = EntropyStrategy().rank([non_candidate, candidate])
        assert ranked[0].guess == "cs"


@pytest.mark.slow
class TestEntropyStrategyRealWordList:
    def test_round_one_matches_game_golden_value(self):
        with open(REPO_ROOT / "src" / "data" / "words.txt") as fp:
            wl = parse_file(fp)
        guesses = sorted(set(wl.target) | set(wl.extra))
        targets = wl.target

        results = analysis.analyze_all(guesses, targets, use_cache=True)
        ranked = EntropyStrategy().rank(results)
        assert ranked[0].guess == "tarse"
        assert ranked[0].entropy == pytest.approx(5.895057463477305)


class TestEntropyStrategyWeighted:
    # Same hand-built pool as TestWeightedAnalyze in test_analysis.py: "dc"
    # finely splits low-weight chaff (higher raw entropy) but lumps the two
    # high-weight plausible answers together; "bb" does the reverse.
    TARGETS = ["aa", "ab", "ca", "cb", "cc", "cd"]
    WEIGHTS = {
        "aa": 100.0,
        "ab": 100.0,
        "ca": 0.001,
        "cb": 0.001,
        "cc": 0.001,
        "cd": 0.001,
    }

    def test_weighted_ranking_prefers_the_guess_that_separates_likely_answers(self):
        results = analysis.analyze_all(
            ["dc", "bb"], self.TARGETS, weights=self.WEIGHTS
        )
        assert EntropyStrategy(weighted=False).rank(results)[0].guess == "dc"
        assert EntropyStrategy(weighted=True).rank(results)[0].guess == "bb"

    def test_weighted_tie_break_prefers_higher_solution_probability(self):
        # Hand-built rather than derived from real word scoring: three
        # analyses tied on weighted_entropy, only two of which are
        # candidates, with candidate_high's solution_probability higher
        # than candidate_low's. Constructing GuessAnalysis directly (rather
        # than searching for real words that happen to tie exactly) is what
        # makes this deterministic -- exact float ties are otherwise hard
        # to hit reliably via real bucket scoring.
        non_candidate = _analysis(
            "xy", entropy=1.0, weighted_entropy=1.0, is_possible_solution=False
        )
        candidate_low = _analysis(
            "lo",
            entropy=1.0,
            weighted_entropy=1.0,
            is_possible_solution=True,
            solution_probability=0.2,
        )
        candidate_high = _analysis(
            "hi",
            entropy=1.0,
            weighted_entropy=1.0,
            is_possible_solution=True,
            solution_probability=0.8,
        )

        # Weighted: scans the whole tied block and picks the highest
        # solution_probability among candidates, not the first one found.
        ranked = EntropyStrategy(weighted=True).rank(
            [non_candidate, candidate_low, candidate_high]
        )
        assert ranked[0].guess == "hi"

        # Unweighted: stops at the first candidate found in the tied block
        # (list order preserved for ties), matching find_best_guess.
        ranked = EntropyStrategy(weighted=False).rank(
            [non_candidate, candidate_low, candidate_high]
        )
        assert ranked[0].guess == "lo"


class TestExpectedPoolSizeStrategy:
    def test_minimizes_expected_remaining_pool_size(self):
        # "aa" splits ["aa","ab","ac"] into a 1-word and a 2-word bucket
        # (E = (1*1 + 2*2)/3); "ax" reveals nothing, so its one bucket
        # covers all 3 (E = 3). "aa" should win.
        results = analysis.analyze_all(["ax", "aa"], ["aa", "ab", "ac"])
        ranked = ExpectedPoolSizeStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_weighted_mode_can_pick_a_different_winner_than_uniform(self):
        # Hand-built: "small_raw" has the smaller raw expected_size but a
        # larger weighted_expected_size than "small_weighted", and vice
        # versa -- exactly the disagreement weighted=True exists to catch.
        small_raw = _analysis("sr", expected_size=1.0, weighted_expected_size=9.0)
        small_weighted = _analysis("sw", expected_size=2.0, weighted_expected_size=1.0)

        assert (
            ExpectedPoolSizeStrategy(weighted=False).rank([small_raw, small_weighted])[
                0
            ].guess
            == "sr"
        )
        assert (
            ExpectedPoolSizeStrategy(weighted=True).rank([small_raw, small_weighted])[
                0
            ].guess
            == "sw"
        )


class TestNumBinsStrategy:
    def test_maximizes_distinct_bucket_count(self):
        # "aa" splits ["aa","ab","ac"] into a 1-word and a 2-word bucket --
        # 2 distinct buckets. "ax" reveals nothing -- 1 bucket. "aa" wins.
        results = analysis.analyze_all(
            ["ax", "aa"], ["aa", "ab", "ac"], include_bucket_stats=True
        )
        ranked = NumBinsStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_prefers_more_bins_over_higher_entropy(self):
        # Hand-built: "manybins" splits into 5 uneven buckets (96/1/1/1/1)
        # -- low entropy (~0.32 bits) despite having more buckets than
        # "fewbins", which splits into 3 near-even buckets (34/33/33) and
        # has much higher entropy (~1.58 bits). NumBinsStrategy should
        # still prefer manybins: entropy rewards *even* splits, this
        # strategy only counts how many buckets exist at all -- that's
        # the whole point of it being a different heuristic from entropy,
        # not just entropy with extra steps.
        manybins = _analysis(
            "manybins", entropy=0.322, bucket_counts=((0, 96), (1, 1), (2, 1), (3, 1), (4, 1))
        )
        fewbins = _analysis(
            "fewbins", entropy=1.585, bucket_counts=((0, 34), (1, 33), (2, 33))
        )
        ranked = NumBinsStrategy().rank([fewbins, manybins])
        assert ranked[0].guess == "manybins"

    def test_ties_on_bin_count_broken_by_entropy(self):
        low_entropy = _analysis(
            "lo", entropy=1.0, bucket_counts=((0, 8), (1, 1), (2, 1))
        )
        high_entropy = _analysis(
            "hi", entropy=1.5, bucket_counts=((0, 5), (1, 4), (2, 1))
        )
        ranked = NumBinsStrategy().rank([low_entropy, high_entropy])
        assert ranked[0].guess == "hi"

    def test_ties_prefer_a_possible_solution(self):
        non_candidate = _analysis(
            "xy", entropy=1.0, bucket_counts=((0, 5), (1, 5)), is_possible_solution=False
        )
        candidate = _analysis(
            "cd", entropy=1.0, bucket_counts=((0, 5), (1, 5)), is_possible_solution=True
        )
        ranked = NumBinsStrategy().rank([non_candidate, candidate])
        assert ranked[0].guess == "cd"

    def test_falls_back_to_buckets_when_bucket_counts_is_unset(self):
        # analyze() (as opposed to analyze_all) always populates .buckets
        # but only sometimes .bucket_counts -- NumBinsStrategy._num_bins
        # must derive bin count from .buckets in that case, matching
        # TwoPlyExpectimaxStrategy._counts_for's fallback.
        a = GuessAnalysis(
            guess="fb",
            buckets={0: ["aa"], 1: ["ab", "ac"]},
            entropy=1.0,
            worst_case_size=2,
            expected_size=1.5,
            is_possible_solution=False,
        )
        assert NumBinsStrategy._num_bins(a) == 2

    def test_requires_bucket_stats_flag_is_set(self):
        assert NumBinsStrategy.requires_bucket_stats is True


@pytest.mark.slow
class TestNumBinsStrategyRealWordList:
    def test_round_one_matches_measured_golden_value(self):
        # "salet" is also the well-known Selby-optimal opener for the
        # original (uniform, unweighted) NYT Wordle list -- a reassuring
        # sign this heuristic surfaces genuinely strong openers, not an
        # artifact of this repo's specific word list.
        with open(REPO_ROOT / "src" / "data" / "words.txt") as fp:
            wl = parse_file(fp)
        guesses = sorted(set(wl.target) | set(wl.extra))
        targets = wl.target

        results = analysis.analyze_all(guesses, targets, use_cache=True, include_bucket_stats=True)
        ranked = NumBinsStrategy().rank(results)
        assert ranked[0].guess == "salet"
        assert len(ranked[0].bucket_counts) == 161


class TestMaxBinsBalanceStrategy:
    def test_prefers_matching_the_round_max_bucket_count_over_being_even_on_its_own(self):
        # Pool of 100. "few" splits into 2 perfectly-even buckets (50/50)
        # -- flawless relative to ITS OWN bucket count, which is exactly
        # the degenerate measure this strategy deliberately avoids. "many"
        # splits into 4 buckets (30/30/20/20) -- less even, but it's also
        # the guess that sets k_target=4 this round, so it's compared
        # against a uniform-over-4 target instead of uniform-over-2.
        # EMD: few=100.0, many=20.0 (hand-verified against
        # _emd_from_uniform directly before writing this assertion).
        few = _analysis("aabb", entropy=1.0, bucket_counts=((0, 50), (1, 50)))
        many = _analysis(
            "ccdd", entropy=1.9, bucket_counts=((0, 30), (1, 30), (2, 20), (3, 20))
        )
        ranked = MaxBinsBalanceStrategy().rank([few, many])
        assert ranked[0].guess == "ccdd"

    def test_not_degenerate_for_a_single_bucket_guess(self):
        # Regression check for the exact failure mode this design avoids
        # (see class docstring): "useless" produces just 1 bucket
        # (uninformative -- every candidate scores identically), which
        # would trivially score a perfect 0 against a uniform-over-1
        # target. Against the round's real k_target=3 (set by "useful",
        # which splits the pool into 3 perfectly-even buckets), it scores
        # a clearly worse 60.0 instead of tying with "useful"'s 0.0.
        useless = _analysis("aaaa", entropy=0.0, bucket_counts=((0, 60),))
        useful = _analysis(
            "bcde", entropy=1.58, bucket_counts=((0, 20), (1, 20), (2, 20))
        )
        ranked = MaxBinsBalanceStrategy().rank([useless, useful])
        assert ranked[0].guess == "bcde"

    def test_weighted_mode_uses_masses_not_counts(self):
        # "p" is perfectly even by raw count (2/2) but very uneven by
        # mass (1.0/9.0); "q" is the reverse (1/3 by count, 5.0/5.0 by
        # mass). Unweighted should prefer p (count_emd 0 vs 1); weighted
        # should prefer q (mass_emd 0 vs 4) -- exactly the disagreement
        # weighted=True exists to catch, same shape as
        # TestEntropyStrategyWeighted's "dc"/"bb" pool.
        p = _analysis(
            "p", entropy=1.0, bucket_counts=((0, 2), (1, 2)), bucket_masses=((0, 1.0), (1, 9.0))
        )
        q = _analysis(
            "q", entropy=0.8, bucket_counts=((0, 1), (1, 3)), bucket_masses=((0, 5.0), (1, 5.0))
        )
        assert MaxBinsBalanceStrategy(weighted=False).rank([p, q])[0].guess == "p"
        assert MaxBinsBalanceStrategy(weighted=True).rank([p, q])[0].guess == "q"

    def test_ties_broken_by_entropy(self):
        low_entropy = _analysis("lo", entropy=1.0, bucket_counts=((0, 5), (1, 5)))
        high_entropy = _analysis("hi", entropy=1.5, bucket_counts=((0, 5), (1, 5)))
        ranked = MaxBinsBalanceStrategy().rank([low_entropy, high_entropy])
        assert ranked[0].guess == "hi"

    def test_ties_prefer_a_possible_solution(self):
        non_candidate = _analysis(
            "xy", entropy=1.0, bucket_counts=((0, 5), (1, 5)), is_possible_solution=False
        )
        candidate = _analysis(
            "cd", entropy=1.0, bucket_counts=((0, 5), (1, 5)), is_possible_solution=True
        )
        ranked = MaxBinsBalanceStrategy().rank([non_candidate, candidate])
        assert ranked[0].guess == "cd"

    def test_weighted_tie_break_prefers_higher_solution_probability(self):
        non_candidate = _analysis(
            "xy", entropy=1.0, bucket_counts=((0, 5), (1, 5)), is_possible_solution=False
        )
        candidate_low = _analysis(
            "lo",
            entropy=1.0,
            bucket_counts=((0, 5), (1, 5)),
            is_possible_solution=True,
            solution_probability=0.2,
        )
        candidate_high = _analysis(
            "hi",
            entropy=1.0,
            bucket_counts=((0, 5), (1, 5)),
            is_possible_solution=True,
            solution_probability=0.8,
        )
        ranked = MaxBinsBalanceStrategy(weighted=True).rank(
            [non_candidate, candidate_low, candidate_high]
        )
        assert ranked[0].guess == "hi"

    def test_falls_back_to_buckets_when_bucket_counts_is_unset(self):
        a = GuessAnalysis(
            guess="fb",
            buckets={0: ["aa"], 1: ["ab", "ac"]},
            entropy=1.0,
            worst_case_size=2,
            expected_size=1.5,
            is_possible_solution=False,
        )
        assert MaxBinsBalanceStrategy._counts_for(a) == ((0, 1), (1, 2))

    def test_handles_empty_analyses_list(self):
        assert MaxBinsBalanceStrategy().rank([]) == []

    def test_requires_bucket_stats_flag_is_set(self):
        assert MaxBinsBalanceStrategy.requires_bucket_stats is True


@pytest.mark.slow
class TestMaxBinsBalanceStrategyRealWordList:
    @classmethod
    def setup_class(cls):
        with open(REPO_ROOT / "src" / "data" / "words.txt") as fp:
            wl = parse_file(fp)
        cls.guesses = sorted(set(wl.target) | set(wl.extra))
        cls.targets = wl.target
        cls.weights = wl.weights

    def test_round_one_matches_measured_golden_value_unweighted(self):
        results = analysis.analyze_all(
            self.guesses, self.targets, use_cache=True, include_bucket_stats=True
        )
        ranked = MaxBinsBalanceStrategy().rank(results)
        assert ranked[0].guess == "tarse"

    def test_round_one_matches_measured_golden_value_weighted(self):
        results = analysis.analyze_all(
            self.guesses,
            self.targets,
            weights=self.weights,
            use_cache=True,
            include_bucket_stats=True,
        )
        ranked = MaxBinsBalanceStrategy(weighted=True).rank(results)
        assert ranked[0].guess == "tarse"

    def test_solves_the_known_hard_cluster_words_within_six_guesses(self):
        # "fazed"/"hazed" are part of the same large, mostly floor-weighted
        # "?a?ed" cluster that broke EntropyStrategy(weighted=True) and an
        # unwrapped TwoPlyExpectimaxStrategy(weighted=True) earlier this
        # session (see the ExactEndgameStrategy/endgame.py work, since
        # reverted). Regression lock, not a coincidence check.
        for weighted in [False, True]:
            engine = SolverEngine(
                self.guesses,
                self.targets,
                MaxBinsBalanceStrategy(weighted=weighted),
                weights=self.weights,
            )
            for solution in ["fazed", "hazed"]:
                engine.reset()
                for _ in range(6):
                    suggestion = engine.suggest()
                    score = scoring.get_score(suggestion.guess, solution)
                    result = engine.apply_score(suggestion.guess, score)
                    if result.outcome == RoundOutcome.SOLVED:
                        break
                    assert result.outcome == RoundOutcome.CONTINUE
                else:
                    pytest.fail(f"weighted={weighted}: {solution} not solved within 6 guesses")


class TestMinimaxStrategy:
    def test_minimizes_worst_case_bucket_size(self):
        # "ax" reveals nothing -> one bucket of size 3 (worst case 3).
        # "aa" splits into buckets of size 1 and 2 (worst case 2).
        results = analysis.analyze_all(["ax", "aa"], ["aa", "ab", "ac"])
        ranked = MinimaxStrategy().rank(results)
        assert ranked[0].guess == "aa"
        assert ranked[0].worst_case_size == 2


class TestTwoPlyExpectimaxStrategy:
    def test_ranks_guesses_by_two_ply_expectimax(self):
        guesses, targets = ["ax", "aa"], ["aa", "ab", "ac"]
        results = analysis.analyze_all(guesses, targets, include_bucket_stats=True)
        ranked = TwoPlyExpectimaxStrategy().rank(results)
        assert ranked[0].guess == "aa"

    def test_handles_empty_analyses_list(self):
        assert TwoPlyExpectimaxStrategy().rank([]) == []

    def test_weighted_ranking_uses_weights(self):
        targets = ["aa", "ab", "ca", "cb"]
        weights = {"aa": 100.0, "ab": 100.0, "ca": 0.001, "cb": 0.001}
        guesses = ["aa", "ca", "xx"]
        results = analysis.analyze_all(
            guesses, targets, weights=weights, include_bucket_stats=True
        )
        ranked = TwoPlyExpectimaxStrategy(weighted=True).rank(results)
        assert len(ranked) == len(results)
        assert ranked[0].guess == "aa"

    def test_requires_bucket_stats_flag_is_set(self):
        assert TwoPlyExpectimaxStrategy.requires_bucket_stats is True


class TestDecisionTreeStrategy:
    def test_decision_tree_strategy_with_dict(self):
        from strategies import DecisionTreeStrategy

        tree = {
            "tree": {
                "guess": "tarse",
                "branches": {
                    "242": {"guess": "tarse", "leaf": True},
                    "0": {
                        "guess": "colin",
                        "branches": {
                            "242": {"guess": "colin", "leaf": True},
                        },
                    },
                },
            }
        }
        targets = ["tarse", "colin"]
        strat = DecisionTreeStrategy(tree_source=tree, target_list=targets)

        # At root with both targets
        analyses = analysis.analyze_all(["tarse", "colin", "abcde"], targets)
        ranked = strat.rank(analyses)
        assert ranked[0].guess == "tarse"

        # At branch with only "colin"
        analyses_colin = analysis.analyze_all(["tarse", "colin", "abcde"], ["colin"])
        ranked_colin = strat.rank(analyses_colin)
        assert ranked_colin[0].guess == "colin"
