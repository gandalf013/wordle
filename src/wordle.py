#!/usr/bin/env python
"""Thin entry-point shim. All real logic lives in scoring.py, fast_scoring.py,
wordlists.py, analysis.py, strategies.py, engine.py, and cli.py.
"""

from cli import main

if __name__ == "__main__":
    main()
