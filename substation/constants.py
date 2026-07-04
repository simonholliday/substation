"""
Global constants for Substation.

This module contains configuration constants used throughout the application.
These are separated from runtime configuration (config.yaml) because they are
typically not changed by users and represent technical parameters tuned for
specific algorithms and hardware characteristics.
"""

# ==============================================================================
# NFM (Narrow FM) Demodulation Constants
# ==============================================================================

# De-emphasis time constant (τ = RC time constant of high-pass filter)
# Transmitters pre-emphasize high frequencies to improve SNR, receivers must de-emphasize
# 300µs is the standard for narrow FM (PMR, amateur radio, etc.)
# Different from broadcast FM which uses 75µs (USA) or 50µs (Europe)
NFM_DEEMPHASIS_TAU = 300e-6  # 300 microseconds

# Maximum frequency deviation for NFM
# Used to normalize the demodulated audio to the range [-1, 1]
# NFM typically uses ±2.5 kHz deviation (narrow compared to broadcast FM's ±75 kHz)
NFM_DEVIATION_HZ = 2.5e3  # 2.5 kHz peak deviation

# Oversampling factor for Intermediate Frequency (IF) decimation
# A factor of 3 to 4 times the final audio rate ensures the FM discriminator 
# has enough bandwidth to capture the full signal deviation and maintain linearity.
NFM_IF_OVERSAMPLE = 4.0

# ==============================================================================
# AM (Amplitude Modulation) Demodulation Constants
# ==============================================================================

# Oversampling factor for Intermediate Frequency (IF) decimation in AM.
# Currently 4.0, matching NFM_IF_OVERSAMPLE, but kept as a separate
# constant because the *reason* is different: for AM the IF stage just
# needs enough headroom above the audio Nyquist for the envelope to track
# without aliasing (AM voice is ~5 kHz wide, so 64 kHz IF is generous).
# Having its own name means you can tune AM and NFM independently in the
# future without the change silently affecting the other demodulator.
AM_IF_OVERSAMPLE = 4.0

# Automatic Gain Control (AGC) compensates for varying signal strengths
# Too-fast AGC sounds "pumpy", too-slow AGC doesn't adapt quickly enough

# Minimum AGC gain level (floor)
# Prevents excessive amplification of noise when signal is very weak
AM_AGC_FLOOR = 0.02  # 2% minimum gain

# AGC attack time: how quickly gain decreases when signal gets stronger
# Fast attack prevents distortion from sudden loud signals
AM_AGC_ATTACK_MS = 10.0  # 10 milliseconds

# AGC release time: how quickly gain increases when signal gets weaker
# Slow release sounds more natural (avoids "pumping" artifacts)
AM_AGC_RELEASE_MS = 200.0  # 200 milliseconds

# Post-AGC output gain scaling
# AM demodulation can produce peaks, so we scale down to prevent clipping
AM_OUTPUT_GAIN = 0.5  # 50% (-6 dB)

# ==============================================================================
# SSB (Single Sideband) Demodulation Constants
# ==============================================================================

# Center frequency of the SSB voice audio band, in Hz.
# ITU/amateur SSB voice occupies roughly 300 - 2700 Hz of audio bandwidth, so
# the spectral center of a transmission sits around 1500 Hz.  The SSB
# demodulator shifts the wanted sideband down by this amount to center the
# audio on DC, where the post-shift low-pass filter can clip the unwanted
# sideband symmetrically.
SSB_AUDIO_CENTER_HZ = 1500.0

# Half-bandwidth of the SSB audio low-pass filter, in Hz.  After the
# frequency shift the audio runs from -1500 Hz to +1500 Hz; this filter
# rejects everything outside that band, including the unwanted sideband
# (which is now sitting on the negative-frequency side of DC).
SSB_AUDIO_HALF_BW_HZ = 1500.0

# IIR filter order for the SSB sideband-rejection low-pass.
# After the Weaver shift, the unwanted sideband sits ~2x the LPF cutoff
# above DC (a 1 kHz tone in the wrong sideband ends up at 2.5 kHz when
# the cutoff is 1.5 kHz).  Order 5 gives only ~22 dB rejection there,
# which the downstream AGC then partially undoes by amplifying the
# residual.  Order 8 gives ~36 dB and is well within scipy's numerical
# stability range for sosfilt — chosen as a balance between rejection
# and transient ringing on voice content.
SSB_LPF_ORDER = 8

# ==============================================================================
# Channel Detection and Scanning Constants
# ==============================================================================

# Fraction of channel spacing to use as channel bandwidth
# For example, with 12.5 kHz spacing: channel width = 12.5 * 0.84 = 10.5 kHz
# Leaves a small guard band (0.16 * spacing) between channels to reduce crosstalk
# 0.84 is empirically chosen to balance channel separation vs signal capture
CHANNEL_WIDTH_FRACTION = 0.84

# Number of FFT bins to exclude around DC (0 Hz offset from center frequency)
# Most SDR receivers have a DC spike caused by LO leakage and I/Q imbalance
# Excluding ±3 bins typically removes the spike without losing too much signal
# For a 2 MHz sample rate with 8192-bin FFT: ±3 bins = ±732 Hz excluded
DC_SPIKE_BINS = 3

# Number of overlapping segments for Welch's method of PSD estimation
# More segments = lower variance (smoother PSD) but lower frequency resolution
# 8 segments provides good balance: 50% overlap gives 15 independent estimates
# Higher values reduce noise but make narrowband signals harder to distinguish
WELCH_SEGMENTS = 8

# Minimum power variance (in dB) across segments for a channel to be considered
# active.  Used to reject stationary noise that crosses the SNR threshold but
# contains no real signal.  Voice and data signals fluctuate substantially over
# time as syllables, frames, or bursts come and go (typically 5-15 dB swings
# within a 200 ms slice).  Stationary noise produces variance close to the
# natural sampling variance of an 8-segment Welch PSD (~1-2 dB).  A threshold
# of 3 dB cleanly separates the cases.
ACTIVATION_VARIANCE_DB = 3.0

# Demodulated audio RMS below which a channel is considered "silent."
# Used by the audio silence timeout to stop recording when the transmitter
# is keyed but not speaking (common on AM airband).  The soft limiter
# normalises output levels, so 0.01 is stable across bands and gain settings.
AUDIO_SILENCE_RMS_THRESHOLD = 0.01

# Spectral flatness (Wiener entropy) threshold for noise rejection.
# Noise has a flat spectrum (flatness 0.3-0.5); any real signal — voice,
# data, tones — has a peaked spectrum (flatness < 0.04).  0.15 sits in
# the large gap between the two groups.  Used by Gate 2 (turn-ON
# speculative demod check) and Gate 3b (post-recording whole-file check).
SPECTRAL_FLATNESS_THRESHOLD = 0.15

# Carrier transient detection threshold.  AM transmitters produce sharp
# clicks when keying on/off.  A carrier transient must exceed
# CARRIER_TRANSIENT_RATIO × the local noise floor, and the surrounding
# checks in recording.py additionally require the quiet region on the
# spike's far side to stay below 25% of the spike peak.  Voice transients
# (plosive consonants) fail that silence criterion because they are
# surrounded by other voice content.
CARRIER_TRANSIENT_RATIO = 8.0

# ==============================================================================
# Noise Floor Estimation Constants
# ==============================================================================

# EMA (Exponential Moving Average) smoothing factor for the noise floor estimate.
# Lower values = more smoothing (slower to adapt).  0.15 provides a ~1 second
# settling time at typical slice rates (~6-10 slices/sec) while filtering out
# per-slice jitter from adjacent-channel activity and SDR gain fluctuations.
NOISE_FLOOR_EMA_ALPHA = 0.15

# Number of processing slices to absorb before enabling detection.
# SDR hardware (especially RTL-SDR) produces transient spikes at startup from
# PLL settling and AGC convergence.  10 slices at ~100ms each ≈ 1 second.
NOISE_FLOOR_WARMUP_SLICES = 10

# ==============================================================================
# Sample-Level Transition Trimming Constants
# ==============================================================================

# Amplitude threshold (0-1 linear scale) for sample-level transition refinement.
# After coarse PSD-based transition detection, the demodulated audio is scanned
# to find the exact sample where signal begins/ends.  This threshold should be
# well above the noise floor but below typical signal levels.
TRIM_AMPLITUDE_THRESHOLD = 0.02

# Number of audio samples to keep as padding before signal onset (fade-in region)
# and after signal end (fade-out region).  The fade is applied only to this
# padding, preserving the full signal content including attack transients.
# At 16 kHz audio: 240 samples ≈ 15ms, 800 samples ≈ 50ms.
TRIM_PRE_SAMPLES = 240
TRIM_POST_SAMPLES = 800

# ==============================================================================
# CTCSS / DCS Subaudible Tone Constants
# ==============================================================================

# The 51 standard CTCSS (Continuous Tone-Coded Squelch System) frequencies
# in Hz, from EIA/TIA-603.  These subaudible tones (67-254 Hz) are
# transmitted alongside voice on NFM channels to identify talk groups.
CTCSS_TONES = (
	67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5,
	94.8, 97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3,
	131.8, 136.5, 141.3, 146.2, 150.0, 151.4, 156.7, 159.8, 162.2, 165.5,
	167.9, 171.3, 173.8, 177.3, 179.9, 183.5, 186.2, 189.9, 192.8, 196.6,
	199.5, 203.5, 206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3,
	254.1,
)

# DCS (Digital-Coded Squelch) bitrate in bps.  DCS transmits a continuous
# 134.3 bps FSK signal below 300 Hz, encoding a 23-bit Golay(23,12) code
# word containing a 9-bit talk group code (octal 000-777).
DCS_BITRATE = 134.3

# The 104 standard DCS codes (octal) from the TIA-603 tables that radio
# manufacturers actually use.  Because the DCS bitstream repeats with no
# frame marker and Golay(23,12) is a cyclic code, rotated alignments of a
# transmission also decode as valid-looking codes — the standard list
# contains one representative per rotation class precisely so receivers
# can pick the intended code.  detect_dcs() only accepts codes from this
# list: transmissions of a standard code decode exactly, transmissions of
# a non-standard "equivalent" code normalise onto their standard-table
# representative (the same behaviour as commercial receivers), and random
# noise decodes are rejected.
DCS_STANDARD_CODES = frozenset((
	0o023, 0o025, 0o026, 0o031, 0o032, 0o036, 0o043, 0o047, 0o051, 0o053,
	0o054, 0o065, 0o071, 0o072, 0o073, 0o074, 0o114, 0o115, 0o116, 0o122,
	0o125, 0o131, 0o132, 0o134, 0o143, 0o145, 0o152, 0o155, 0o156, 0o162,
	0o165, 0o172, 0o174, 0o205, 0o212, 0o223, 0o225, 0o226, 0o243, 0o244,
	0o245, 0o246, 0o251, 0o252, 0o255, 0o261, 0o263, 0o265, 0o266, 0o271,
	0o274, 0o306, 0o311, 0o315, 0o325, 0o331, 0o332, 0o343, 0o346, 0o351,
	0o356, 0o364, 0o365, 0o371, 0o411, 0o412, 0o413, 0o423, 0o431, 0o432,
	0o445, 0o446, 0o452, 0o454, 0o455, 0o462, 0o464, 0o465, 0o466, 0o503,
	0o506, 0o516, 0o523, 0o526, 0o532, 0o546, 0o565, 0o606, 0o612, 0o624,
	0o627, 0o631, 0o632, 0o654, 0o662, 0o664, 0o703, 0o712, 0o723, 0o731,
	0o732, 0o734, 0o743, 0o754,
))

# NFM voice bandpass filter limits (Hz).  Applied after CTCSS/DCS detection
# to remove subaudible tones from the recording while preserving voice.
# 300 Hz removes CTCSS (67-254 Hz) and DCS (<300 Hz).
# 3400 Hz is the standard voice telephony upper limit.
NFM_VOICE_HIGHPASS_HZ = 300.0
NFM_VOICE_LOWPASS_HZ = 3400.0
