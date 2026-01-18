"""
SDR Scanner - Software-defined radio scanning package

This package provides tools for scanning and recording radio bands using SDR devices.

Public API:
	RadioScanner: Access via sdr_scanner.scanner.scanner.RadioScanner
	ChannelRecorder: Access via sdr_scanner.recording.recorder.ChannelRecorder

Example:
	import sdr_scanner.scanner.scanner
	scanner = sdr_scanner.scanner.scanner.RadioScanner(...)
"""

__version__ = "1.0.0"

__all__ = [
	'__version__',
]
