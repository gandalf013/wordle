"""Tests for wordle.py, now a thin entry-point shim. All real behavior is
tested where it actually lives: test_scoring.py, test_fast_scoring.py,
test_wordlists.py, test_analysis.py, test_strategies.py, test_engine.py,
and test_cli.py.
"""

from cli import main as cli_main

import wordle


def test_main_delegates_to_cli_main():
    assert wordle.main is cli_main
