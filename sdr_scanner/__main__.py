"""
Entry point for running sdr_scanner as a module: python -m sdr_scanner
"""

import sys
import sdr_scanner.cli


if __name__ == '__main__':
	sys.exit(sdr_scanner.cli.main())
