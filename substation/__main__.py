"""
Entry point for running Substation as a module.

This file allows the package to be executed as:
    python -m substation [args]

It simply delegates to the CLI main() function which handles
all argument parsing and program execution.
"""

import sys
import substation.cli

if __name__ == '__main__':
	sys.exit(substation.cli.main())
