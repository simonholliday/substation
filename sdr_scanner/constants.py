"""
Global constants for SDR Scanner
"""

# NFM demodulation constants
NFM_DEEMPHASIS_TAU = 300e-6  # 300µs time constant (standard for NFM)
NFM_DEVIATION_HZ = 5000      # Max frequency deviation for normalization

# AM AGC constants
AM_AGC_ALPHA = 0.05          # AGC smoothing factor (lower = slower response)

# RadioScanner class constants
# Hysteresis margin in dB - channel turns ON at threshold, OFF at threshold minus HYSTERESIS_DB
HYSTERESIS_DB = 3.0

# Fraction of channel spacing used as default channel width
CHANNEL_WIDTH_FRACTION = 0.84

# Number of DC bins to exclude around center frequency (RTL-SDR DC spike)
DC_SPIKE_BINS = 3

# Number of FFT segments for Welch averaging
WELCH_SEGMENTS = 8
