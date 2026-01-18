"""
Backwards compatibility wrapper for main.py

This file provides backwards compatibility for existing workflows that use:
    python main.py

All functionality has been moved to the sdr_scanner package.
For new workflows, use:
    python -m sdr_scanner
or after installation:
    sdr-scanner
"""

import sys
import sdr_scanner.cli

if __name__ == '__main__':
	sys.exit(sdr_scanner.cli.main())
