"""
Type definitions and data classes for SDR Scanner
"""

import dataclasses


@dataclasses.dataclass
class ChannelSpec:
	"""
	Specification for a radio channel

	Attributes:
		band_name: Name of the band (e.g., 'pmr', 'marine')
		channel_index: Channel index number
		channel_freq: Channel center frequency in Hz
		modulation: Modulation type (e.g., 'NFM', 'AM')
	"""
	band_name: str
	channel_index: int
	channel_freq: float
	modulation: str = "Unknown"
