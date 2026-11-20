#!/usr/bin/env python3
"""Main console entrypoint for the Autonomous Chess Game Recording System.

Dispatches command-line arguments to subcommands: `record`, `audit`, `replay`, `benchmark`.
"""

import sys
from src.ui.cli import main

if __name__ == "__main__":
    sys.exit(main())

