#!/usr/bin/env python

import argparse
import logging
import math
import sys
import collections
from enum import IntEnum

import numpy as np
from scipy.stats import entropy as get_entropy
from tqdm import tqdm


class Score(IntEnum):
    GRAY = 0
    YELLOW = 1
    GREEN = 2


SQUARES = {
    Score.GRAY: "⬛",
    Score.YELLOW: "🟨",
    Score.GREEN: "🟩",
}

GRAYS = [Score.GRAY for _ in range(5)]

GRAY = Score.GRAY
YELLOW = Score.YELLOW
GREEN = Score.GREEN


class GameState(IntEnum):
    QUIT = 0
    RESTART = 1
    CONTINUE = 2
    ERROR = 3
    SOLVED = 4


class Game:
    def __init__(
        self,
        guess_list,
        target_list,
        solution=None,
        automatic=False,
        initial_guess=None,
        threshold_display=3,
        num_top_guesses=1,
    ):
        self.guess_list = np.array(guess_list)
        self.target_lists = [np.array(target_list)]
        self.solution = solution
        self.automatic = automatic
        self.n = len(self.guess_list[0])
        self.found_solution = None
        self.initial_guess = initial_guess
        self.best_initial_guess = None
        self.threshold_display = threshold_display
        self.num_top_guesses = num_top_guesses
        self._scores = []

    @property
    def round(self):
        return len(self.target_lists) - 1

    def get_score(self, guess, target):
        if len(guess) != len(target):
            raise ValueError(f"Guess {guess} not valid for target {target}")
        if len(guess) != self.n:
            raise ValueError(f"Guess {guess}/{len(guess)} not valid: {self.n}")

        c = collections.Counter(target)
        score = GRAYS[:]
        for i, (g, t) in enumerate(zip(guess, target)):
            if g == t:
                score[i] = Score.GREEN
                c[g] -= 1

        for i, g in enumerate(guess):
            if score[i] != Score.GREEN and c[g]:
                score[i] = Score.YELLOW
                c[g] -= 1

        return self.get_score_num(score)

    def get_score_num(self, score):
        return sum(3 ** (self.n - i - 1) * s for i, s in enumerate(score))

    def get_score_list(self, score):
        r = []
        while score:
            score, rem = divmod(score, 3)
            r.append(Score(rem))

        r.extend([Score.GRAY] * (self.n - len(r)))
        return r[::-1]

    def get_score_str(self, score):
        if isinstance(score, int):
            score = self.get_score_list(score)

        return "".join(SQUARES[s] for s in score)

    def score_guess(self, guess):
        target_list = self.target_lists[self.round]
        return np.array([self.get_score(guess, target) for target in target_list])

    def get_census(self, scores):
        return np.bincount(scores, minlength=3 ** self.n)

    def get_all_censuses(self, limit=None):
        all_censuses = []
        if limit is None:
            limit = len(self.guess_list)

        for i in tqdm(range(limit)):
            scores = self.score_guess(self.guess_list[i])
            census = self.get_census(scores)
            all_censuses.append(census)

        return np.array(all_censuses)

    def get_all_entropy(self, censuses):
        return get_entropy(censuses, axis=1, base=2)

    def find_best_guess(self):
        target_list = set(self.target_lists[self.round])
        censuses = self.get_all_censuses()
        entropy = self.get_all_entropy(censuses)
        eindices = np.argsort(entropy)[::-1]
        best_entropy = entropy[eindices[0]]
        best_guess = self.guess_list[eindices[0]]
        if self.num_top_guesses > 1:
            logging.info(f"Top {self.num_top_guesses} guesses:")
            guesses = self.guess_list[eindices[: self.num_top_guesses]]
            entropies = entropy[eindices[: self.num_top_guesses]]
            for g, e in zip(guesses, entropies):
                logging.info(f"{g} {e}")

        if best_guess not in target_list:
            for i in eindices[1:]:
                e = entropy[i]
                if not math.isclose(e, best_entropy):
                    break

                guess = self.guess_list[i]
                if guess in target_list:
                    logging.info(
                        f"Using {guess} instead of {best_guess} for better chances"
                    )
                    best_guess = guess
                    break

        logging.info(f"Best guess {best_guess} entropy {best_entropy}")
        return best_guess

    def reset(self):
        self.target_lists = [self.target_lists[0]]
        self.found_solution = None
        self._scores = []

    def play_one_round(self):
        if self.found_solution is not None:
            logging.info(f"Already found solution: {self.found_solution}")
            return False

        if self.round == 0 and (
            self.best_initial_guess is not None or self.initial_guess is not None
        ):
            if self.initial_guess is not None:
                logging.info(f"Using supplied initial guess {self.initial_guess}")
                best_guess = self.initial_guess
            else:
                logging.info("Using pre-computed round 1 values")
                best_guess = self.best_initial_guess
        else:
            best_guess = self.find_best_guess()
            if self.round == 0:
                self.best_initial_guess = best_guess

        new_suggestion = ""
        if not self.automatic:
            try:
                new_suggestion = input(f"Suggested {best_guess}. Score/new guess: ")
            except EOFError:
                return GameState.QUIT
            else:
                new_suggestion = new_suggestion.lower().strip()

        if (
            new_suggestion
            and len(new_suggestion) == self.n
            and new_suggestion.islower()
        ):
            logging.info(f"Using {new_suggestion} instead of {best_guess}")
            best_guess, new_suggestion = new_suggestion, None

        state, guess_score = self.get_guess_score(best_guess, new_suggestion)
        if guess_score is not None:
            self._scores.append((best_guess, guess_score))
            sys.stdout.write("%s\n" % self.get_score_str(guess_score))

        if state != GameState.CONTINUE:
            return state

        target_list = self.target_lists[self.round]
        new_target_list = [
            word
            for word in target_list
            if self.get_score(best_guess, word) == guess_score
        ]
        if not new_target_list:
            logging.error("No guess matches the score!")
            return GameState.ERROR

        self.target_lists.append(np.array(new_target_list))
        if len(new_target_list) == 1:
            self.found_solution = new_target_list[0]
            perfect_score = self.get_score_num([2] * self.n)
            if not (self._scores and self._scores[-1][1] == perfect_score):
                self._scores.append((self.found_solution, perfect_score))
            logging.info(
                f"SOLVED: {self.found_solution} in {len(self._scores)} guesses"
            )
            self.display_scores()
            return GameState.SOLVED

        logging.info(f"{len(new_target_list)} words match the pattern")
        if len(new_target_list) <= self.threshold_display:
            logging.info(f"Matching words: {sorted(new_target_list)}")

        logging.debug(f"Words:\n{new_target_list}")
        return GameState.CONTINUE

    def display_scores(self):
        for guess, score in self._scores:
            sys.stdout.write(f"{guess} {self.get_score_str(score)}\n")

    def get_guess_score(self, guess, potential_score=None):
        if self.solution is not None:
            return GameState.CONTINUE, self.get_score(guess, self.solution)

        if potential_score:
            try:
                potential_score = int(potential_score, 3)
            except (TypeError, ValueError):
                logging.info(f"Could not understand {potential_score}, ignoring.")
            else:
                return GameState.CONTINUE, potential_score

        s = ""
        while len(s) != self.n:
            try:
                s = input(f"Enter score for '{guess}': ").strip()
            except EOFError:
                logging.info("Quit")
                return GameState.QUIT, None

            if not s:
                continue

            if s[0].lower() == "r":
                logging.info("Restart")
                return GameState.RESTART, None

            if s[0].lower() == "q":
                logging.info("Quit")
                return GameState.QUIT, None

        return GameState.CONTINUE, int(s, base=3)


def parse_file(fp):
    target = []
    extra = []
    r = target
    wordlen = None
    for line in fp:
        data = line.strip()
        if not data:
            if extra:
                raise ValueError("too many blank lines")
            r = extra
            continue

        if wordlen is None:
            wordlen = len(data)
        elif len(data) != wordlen:
            raise ValueError("Bad length {len(data)}, expected {wordlen}")

        r.append(data)

    if set(target) & set(extra):
        raise ValueError("Target and extra words overlap")

    return target, extra, wordlen


def run(args):
    target, extra, wordlen = parse_file(args.infile)
    logging.info(f"Target {len(target)} extra {len(extra)} wordlen {wordlen}")
    words = sorted(set(target) | set(extra))
    if args.guesses == "all":
        guesses = words
    elif args.guesses == "target":
        guesses = target
    elif args.guesses == "extra":
        guesses = extra
    else:
        raise ValueError(f"Unknown 'guesses': {guesses}")

    if args.targets == "all":
        targets = words
    elif args.targets == "target":
        targets = target
    elif args.targets == "extra":
        targets = extra
    else:
        raise ValueError(f"Unknown 'targets': {args.targets}")

    g = Game(
        guesses,
        targets,
        solution=args.solution,
        automatic=args.automatic,
        initial_guess=args.initial_guess,
        threshold_display=args.threshold_display,
        num_top_guesses=args.num_top_guesses,
    )
    state = GameState.CONTINUE
    while state == GameState.CONTINUE:
        state = g.play_one_round()
        if state == GameState.RESTART:
            g.reset()
            state = GameState.CONTINUE
        elif state == GameState.ERROR or state == GameState.SOLVED:
            try:
                r = input("New round? ").lower().strip()
            except EOFError:
                r = ""

            if r and r in ("r", "1", "y"):
                logging.info("New round")
                g.reset()
                state = GameState.CONTINUE
            else:
                state = GameState.QUIT


def setup_logging(debug):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=lvl, format="%(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action="store_true")
    parser.add_argument(
        "infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument("-g", "--guesses", default="all")
    parser.add_argument("-t", "--targets", default="target")
    parser.add_argument("-i", "--initial-guess", default=None)
    parser.add_argument("-T", "--threshold-display", default=3, type=int)
    parser.add_argument("-s", "--solution", default=None)
    parser.add_argument("-a", "--automatic", action="store_true")
    parser.add_argument("-n", "--num-top-guesses", default=1, type=int)

    args = parser.parse_args()
    setup_logging(args.debug)
    run(args)


if __name__ == "__main__":
    main()
