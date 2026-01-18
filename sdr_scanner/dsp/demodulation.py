"""
Demodulation functions for various modulation types
"""

import numpy
import numpy.typing
import scipy.signal
import typing

import sdr_scanner.constants
import sdr_scanner.dsp.filters


def demodulate_nfm(
	iq_samples: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	audio_sample_rate: int,
	state: dict | None = None
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:
	"""
	Demodulate Narrow FM (NFM) from IQ samples with state preservation

	Args:
		iq_samples: Complex IQ samples (already filtered to channel bandwidth)
		sample_rate: Sample rate of IQ samples in Hz
		audio_sample_rate: Desired output audio sample rate in Hz
		state: Optional state dict with 'last_iq' and 'deemph_zi' for continuous demodulation

	Returns:
		Tuple of (audio_samples, new_state) where new_state contains updated filter state
	"""
	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	# Initialize state if needed
	if state is None:
		state = {}

	# FM demodulation: instantaneous frequency = d(phase)/dt
	if 'last_iq' not in state:
		state['last_iq'] = iq_samples[0]

	iq_with_prev = numpy.concatenate(([state['last_iq']], iq_samples))
	demod = numpy.angle(iq_with_prev[1:] * numpy.conj(iq_with_prev[:-1]))
	state['last_iq'] = iq_samples[-1]

	# De-emphasis filter
	tau = sdr_scanner.constants.NFM_DEEMPHASIS_TAU
	alpha = 1.0 / (1.0 + sample_rate * tau)

	if 'deemph_zi' not in state:
		state['deemph_zi'] = scipy.signal.lfilter_zi([alpha], [1, alpha - 1]) * 0.0

	demod_deemph, state['deemph_zi'] = scipy.signal.lfilter(
		[alpha], [1, alpha - 1], demod, zi=state['deemph_zi']
	)

	# DC removal - subtract block mean
	demod_dc_blocked = demod_deemph - numpy.mean(demod_deemph)

	# Normalize to approximate [-1, 1] range
	demod_normalized = demod_dc_blocked / (2 * numpy.pi * sdr_scanner.constants.NFM_DEVIATION_HZ / sample_rate)

	# Clip to [-1, 1] range
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# Decimate to audio sample rate
	return sdr_scanner.dsp.filters.decimate_audio(demod_normalized, sample_rate, audio_sample_rate, state)


def demodulate_am(
	iq_samples: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	audio_sample_rate: int,
	state: dict | None = None
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:
	"""
	Demodulate Amplitude Modulation (AM) from IQ samples with state preservation

	Args:
		iq_samples: Complex IQ samples (already filtered to channel bandwidth)
		sample_rate: Sample rate of IQ samples in Hz
		audio_sample_rate: Desired output audio sample rate in Hz
		state: Optional state dict for AGC and decimation continuity

	Returns:
		Tuple of (audio_samples, new_state) where new_state contains updated filter state
	"""
	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# AM demodulation - extract magnitude (envelope detection)
	demod = numpy.abs(iq_samples)

	# DC removal - subtract block mean
	demod_dc_blocked = demod - numpy.mean(demod)

	# Smoothed AGC using leaky integrator
	block_level = numpy.percentile(numpy.abs(demod_dc_blocked), 99)

	if 'agc_level' not in state:
		state['agc_level'] = block_level if block_level > 0.01 else 1.0

	state['agc_level'] = sdr_scanner.constants.AM_AGC_ALPHA * block_level + (1 - sdr_scanner.constants.AM_AGC_ALPHA) * state['agc_level']

	if state['agc_level'] > 0.01:
		demod_normalized = demod_dc_blocked / state['agc_level']
	else:
		demod_normalized = demod_dc_blocked

	# Clip to [-1.0, 1.0] range
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# Decimate to audio sample rate
	return sdr_scanner.dsp.filters.decimate_audio(demod_normalized, sample_rate, audio_sample_rate, state)


# Dictionary of available demodulators
DEMODULATORS: dict[str, typing.Callable] = {
	'NFM': demodulate_nfm,
	'AM': demodulate_am,
	# Future demodulators can be added here:
	# 'WFM': demodulate_wfm,
}


def get_demodulator(modulation: str) -> typing.Callable:
	"""
	Get demodulator function for a specific modulation type

	Args:
		modulation: Modulation type (e.g., 'NFM', 'AM')

	Returns:
		Demodulator function

	Raises:
		KeyError: If modulation type is not supported
	"""
	if modulation not in DEMODULATORS:
		available = ', '.join(DEMODULATORS.keys())
		raise KeyError(f"Unsupported modulation '{modulation}'. Available: {available}")

	return DEMODULATORS[modulation]
